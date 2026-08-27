# Datenfluss im ConvResBlock (mit FiLM)

Dieses Diagramm zoomt exakt in deinen `ConvResBlock` (aus `flow_matching_cond_spectral_crossattn.py`) hinein. Es zeigt präzise, wie die räumlichen Trajektorien-Daten und der zeitliche Konditionierungsvektor zusammenfließen, inklusive der Tensor-Dimensionen an jedem Schritt.

Hierbei steht:
- **`B`**: Batch Size
- **`L`**: Sequenzlänge (Anzahl der Wegpunkte, z.B. 20, 10 oder 5)
- **`in_ch`** / **`out_ch`**: Anzahl der Feature-Kanäle (z.B. 128, 256)
- **`cond_dim`**: Größe des Konditionierungsvektors (z.B. 128)

```mermaid
graph TD
    %% Inputs
    X["Input Features (x)<br>Tensor: (B, in_ch, L)"]
    COND["Time Condition (cond)<br>Tensor: (B, cond_dim)"]
    
    subgraph "1. Räumliche Vorverarbeitung (Conv & Norm)"
        C1["Conv1d (kernel_size=3)<br>Tensor: (B, out_ch, L)"]
        GN1["GroupNorm<br>Tensor: (B, out_ch, L)"]
    end
    
    subgraph "2. FiLM Parameter-Berechnung"
        F_PROJ["Linear (film_proj)<br>Zero-Initialized!<br>Tensor: (B, out_ch * 2)"]
        CHUNK["Chunk in 2 Teile<br>und Unsqueeze(-1)"]
        GAMMA["Gamma (γ)<br>Tensor: (B, out_ch, 1)"]
        BETA["Beta (β)<br>Tensor: (B, out_ch, 1)"]
    end
    
    subgraph "3. FiLM Modulation (adaGN)"
        MOD["Modulation:<br>h * (1 + γ) + β<br>Tensor: (B, out_ch, L)"]
    end
    
    subgraph "4. Nachbearbeitung & Residual"
        ACT1["SiLU Aktivierung<br>Tensor: (B, out_ch, L)"]
        C2["Conv1d (kernel_size=3)<br>Tensor: (B, out_ch, L)"]
        GN2["GroupNorm<br>Tensor: (B, out_ch, L)"]
        RES["Residual Addition<br>h + residual(x)"]
    end

    %% Routing
    X --> C1
    C1 --> GN1
    GN1 --> MOD
    
    COND --> F_PROJ
    F_PROJ --> CHUNK
    CHUNK --> GAMMA
    CHUNK --> BETA
    
    GAMMA -.->|Broadcasting über L| MOD
    BETA -.->|Broadcasting über L| MOD
    
    MOD --> ACT1
    ACT1 --> C2
    C2 --> GN2
    GN2 --> RES
    
    X -.->|Skip Connection| RES
    
    OUT["Output Features<br>Tensor: (B, out_ch, L)"]
    RES --> OUT

    %% Styling
    classDef feature fill:#e3f2fd,stroke:#1565C0,stroke-width:2px;
    classDef cond fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef film fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,stroke-dasharray: 5 5;
    
    class X,C1,GN1,MOD,ACT1,C2,GN2,OUT feature;
    class COND,F_PROJ,CHUNK,GAMMA,BETA cond;
    class MOD film;
```

### Die Magie liegt im "Broadcasting" (Schritt 3)
Schau dir die Tensor-Dimensionen von Gamma und Beta an: Sie haben die Form `(B, out_ch, 1)`. 
Die räumlichen Features aus der GroupNorm haben die Form `(B, out_ch, L)`.

Da $\gamma$ und $\beta$ in der räumlichen Dimension (Länge) nur die Größe `1` haben, wendet PyTorch **Broadcasting** an: Der exakt selbe Gamma/Beta-Wert wird auf *alle* `L` Wegpunkte dieses Kanals angewendet! 

Genau das meine ich mit "globalem Stil": FiLM steuert den Kanal als Ganzes über die gesamte Sequenz, anstatt für jeden Wegpunkt einen individuellen Wert zu berechnen (letzteres würde Cross-Attention machen).
