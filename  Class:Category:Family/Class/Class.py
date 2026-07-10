# ============================================================
# Ransomware Detection - ransomware.xlsx
# Previsão: Class (Benign vs Malware) — classificação binária
# ============================================================

import re
import warnings
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, recall_score, roc_curve

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ===============================
# 1. Carregar dataset
# ===============================
df = pd.read_excel("ransomware.xlsx")

print(f"\nDataset: {df.shape}")
print("\nDistribuição de Class:")
print(df["Class"].value_counts())

# Label binária: Benign=0, Malware=1
df["label"] = (df["Class"] != "Benign").astype(int)

# ===============================
# 2. Preparar features
# ===============================

# Algumas colunas PE header têm formato "0x00001234 (Section: .text)"
# Esta função extrai apenas o valor hex, ignorando texto extra
def parse_hex(val):
    s = str(val).strip()
    m = re.search(r'0x([0-9a-fA-F]+)', s)
    if m:
        try:
            return int(m.group(1), 16)
        except Exception:
            return 0
    try:
        return int(s, 0)
    except Exception:
        return 0


# Colunas PE header (armazenadas como hex string no xlsx)
hex_cols = [
    "bytes_on_last_page", "pages_in_file", "relocations", "size_of_header",
    "min_extra_paragraphs", "max_extra_paragraphs", "init_ss_value", "init_sp_value",
    "init_ip_value", "init_cs_value", "over_lay_number", "oem_identifier",
    "address_of_ne_header", "Magic", "SizeOfCode", "SizeOfInitializedData",
    "SizeOfUninitializedData", "AddressOfEntryPoint", "BaseOfCode", "BaseOfData",
    "ImageBase", "SectionAlignment", "FileAlignment", "OperatingSystemVersion",
    "ImageVersion", "SizeOfImage", "SizeOfHeaders", "Checksum", "Subsystem",
    "DllCharacteristics", "SizeofStackReserve", "SizeofStackCommit",
    "SizeofHeapCommit", "SizeofHeapReserve", "LoaderFlags",
    "text_VirtualSize", "text_VirtualAddress", "text_SizeOfRawData",
    "text_PointerToRawData", "text_PointerToRelocations", "text_PointerToLineNumbers",
    "text_Characteristics", "rdata_VirtualSize", "rdata_VirtualAddress",
    "rdata_SizeOfRawData", "rdata_PointerToRawData", "rdata_PointerToRelocations",
    "rdata_PointerToLineNumbers", "rdata_Characteristics",
]

for col in hex_cols:
    df[col] = df[col].apply(parse_hex)

# Colunas comportamentais (já numéricas no xlsx)
behavioral_cols = [
    "registry_read", "registry_write", "registry_delete", "registry_total",
    "network_threats", "network_dns", "network_http", "network_connections",
    "processes_malicious", "processes_suspicious", "processes_monitored", "total_procsses",
    "files_malicious", "files_suspicious", "files_text", "files_unknown",
    "dlls_calls", "apis",
]

feature_cols = hex_cols + behavioral_cols  # 67 features no total
print(f"\nTotal de features: {len(feature_cols)}")

X = df[feature_cols].fillna(0)
y = df["label"].values

# Normalização — essencial dada a diferença de escala entre PE header e contagens
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# ===============================
# 3. Dataset e DataLoader PyTorch
# ===============================

class MalwareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, stratify=y, random_state=42
)

# Split de validação para early stopping e calibração do threshold
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

train_dataset = MalwareDataset(X_train, y_train)
val_dataset   = MalwareDataset(X_val,   y_val)
test_dataset  = MalwareDataset(X_test,  y_test)
train_loader  = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=512, shuffle=False)
test_loader   = DataLoader(test_dataset,  batch_size=512, shuffle=False)

print(f"\nTrain: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} amostras")

# ===============================
# 4. Rede Neuronal
# ===============================

class MalwareNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


input_dim = X_train.shape[1]
model     = MalwareNN(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
# Reduz o LR a metade se a val_loss não melhorar em 5 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)

print(f"\nModelo:\n{model}")

# ===============================
# 5. Treinamento com Early Stopping
# ===============================
EPOCHS        = 1000
PATIENCE      = 500       # para se a val_loss não melhorar em 50 epochs
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

Epocas     = []
Losts      = []
Val_Losts  = []

print("\n--- Treinamento ---")
for epoch in range(EPOCHS):
    # -- treino --
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    epoch_loss = total_loss / len(train_dataset)

    # -- validação --
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            out      = model(X_batch)
            val_loss += criterion(out, y_batch).item() * X_batch.size(0)
    val_loss /= len(val_dataset)

    Losts.append(epoch_loss)
    Val_Losts.append(val_loss)
    Epocas.append(epoch + 1)
    print(f"Epoch {epoch+1:>3}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Reduz LR se val_loss estagnar
    scheduler.step(val_loss)

    # -- early stopping --
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping na epoch {epoch+1} (sem melhoria há {PATIENCE} epochs)")
            break

# Restaurar o melhor modelo
model.load_state_dict(best_model_state)
print(f"Melhor modelo restaurado (val_loss={best_val_loss:.4f})")

# ===============================
# 6. Plot Loss Curve
# ===============================
plt.figure(figsize=(10, 5))
plt.plot(Epocas, Losts,     label="Train Loss",      color="blue")
plt.plot(Epocas, Val_Losts, label="Validation Loss", color="orange", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss — Class (Benign vs Malware)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve_class.png", dpi=150)
plt.show()
print("Loss curve guardada em loss_curve_class.png")

# ===============================
# 7. Calibrar threshold no val set
# ===============================
model.eval()
val_probs = []
val_true  = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        probs   = model(X_batch).cpu().numpy()
        val_probs.extend(probs.flatten())
        val_true.extend(y_batch.numpy())

# Encontrar threshold que maximiza F1 macro no conjunto de validação
fpr, tpr, thresholds = roc_curve(val_true, val_probs)
best_thresh = 0.5
best_f1     = 0.0

from sklearn.metrics import f1_score
for t in thresholds:
    preds = (np.array(val_probs) >= t).astype(int)
    f1 = f1_score(val_true, preds, average="macro")
    if f1 > best_f1:
        best_f1     = f1
        best_thresh = t

print(f"\nThreshold ótimo (val set): {best_thresh:.4f}  |  F1 macro val: {best_f1:.4f}")

# ===============================
# 8. Avaliação no test set
# ===============================
all_probs = []
all_preds = []

with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        probs   = model(X_batch).cpu().numpy()
        all_probs.extend(probs.flatten())
        all_preds.extend((probs >= best_thresh).astype(int).flatten())

print("\n=== Resultados — Class (Benign vs Malware) ===")
print(f"Threshold usado: {best_thresh:.4f}  (calibrado no val set)\n")
print(classification_report(y_test, all_preds, target_names=["Benign", "Malware"]))
print(f"AUC:    {roc_auc_score(y_test, all_probs):.4f}")
print(f"Recall: {recall_score(y_test, all_preds):.4f}")

# ===============================
# 9. SHAP — explicabilidade
# ===============================

# Mapeamento de features para descrições legíveis
FEATURE_TEXT = {
    "processes_malicious":   "processos maliciosos detetados",
    "files_malicious":       "ficheiros maliciosos criados",
    "network_threats":       "ameaças de rede detetadas",
    "registry_write":        "escrita suspeita no registo",
    "apis":                  "chamadas de API anómalas",
    "dlls_calls":            "chamadas de DLL suspeitas",
    "processes_suspicious":  "processos suspeitos identificados",
    "network_connections":   "ligações de rede suspeitas",
    "files_suspicious":      "ficheiros suspeitos criados",
    "registry_delete":       "eliminação suspeita no registo",
    "SizeOfCode":            "tamanho de código executável anómalo",
    "SizeOfImage":           "tamanho de imagem PE anómalo",
    "AddressOfEntryPoint":   "entry point PE anómalo",
}

# Amostra pequena para SHAP (evitar timeout)
X_shap = X_train.sample(n=min(10000, len(X_train)), random_state=42)

model.eval()
explainer = shap.Explainer(
    lambda X_np: model(
        torch.tensor(X_np, dtype=torch.float32).to(device)
    ).cpu().detach().numpy().flatten(),
    X_shap.values,
    feature_names=feature_cols,
)


def explain_sample(row_scaled_values, threshold=0.04):
    """Devolve as 3 razões principais para a classificação de uma amostra."""
    shap_values = explainer(row_scaled_values.reshape(1, -1)).values[0]
    impact = sorted(zip(feature_cols, shap_values), key=lambda x: abs(x[1]), reverse=True)
    reasons = []
    for name, val in impact:
        if abs(val) >= threshold and name in FEATURE_TEXT:
            prefix = "" if val > 0 else "baixa evidência de "
            reasons.append(prefix + FEATURE_TEXT[name])
        if len(reasons) >= 3:
            break
    return reasons or ["padrões gerais suspeitos"]


# ===============================
# 10. Análise de amostras individuais (primeiros 20 Benign)
# ===============================
print("\n=== ANALISANDO OS PRIMEIROS 20 FICHEIROS BENIGN ===\n")

sample_df = df[df["Class"] == "Benign"].head(20).copy()
X_sample_raw    = sample_df[feature_cols].fillna(0)
X_sample_scaled = pd.DataFrame(
    scaler.transform(X_sample_raw),
    columns=feature_cols
)

results = []
for i in range(len(sample_df)):
    row        = sample_df.iloc[i]
    row_scaled = X_sample_scaled.iloc[i].values
    x_tensor   = torch.tensor(row_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(x_tensor).item()

    if score >= best_thresh + 0.2:
        action = "BLOCK"
    elif score >= best_thresh:
        action = "WARN"
    else:
        action = "ALLOW"

    true_class = row["Class"]
    reasons    = explain_sample(row_scaled)
    results.append((row["md5"], score, action, true_class, reasons))

# Output formatado
print(f"{'MD5':<34} {'Score':>6}  {'Ação':<6}  {'Real':<8}  Razões")
print("-" * 100)
for md5, score, action, true_class, reasons in results:
    reasons_str = " | ".join(reasons)
    print(f"{md5:<34} {score:.4f}  {action:<6}  {true_class:<8}  {reasons_str}")
