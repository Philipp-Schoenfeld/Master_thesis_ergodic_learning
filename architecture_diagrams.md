# Architektur-Visualisierungen

Hier sind die Diagramme für alle vier Architektur-Iterationen. Sie verdeutlichen, wie die Konditionierung (die Zielform/Verteilung) verarbeitet und in das U-Net injiziert wird, und wo die entscheidenden Engpässe (GAP) bzw. Innovationen (Cross-Attention) sitzen.

### 1. Baseline: Additive Konditionierung (`_char.py`)
Das einfachste Modell. Die Form wird auf einen Vektor zusammengepresst und einfach addiert.
```mermaid
graph TD
    %% Inputs
    X[Rauschende Trajektorie x_t] --> MPD[MPD Layer]
    T[Zeit t] --> TE[Time Embedding]
    S[Zielform] --> SE[ShapeEncoderMPD]
    
    %% Processing Condition
    SE --> GAP((Global Average Pooling))
    GAP --> COND[1D Bedingungs-Vektor]
    
    %% Injection
    TE --> ADD1(+)
    COND --> ADD2(+)
    
    %% U-Net
    MPD --> UNET[1D U-Net Backbone]
    ADD1 --> UNET
    ADD2 -.->|Einfache Addition auf Feature Maps| UNET
    
    %% Output
    UNET --> FLOW[Flow Velocity v_t]
    
    classDef bottleneck fill:#ffcccc,stroke:#ff0000,stroke-width:2px;
    class GAP bottleneck;
```

---

### 2. FiLM Konditionierung (`_char_film.py`)
Ersetzt die einfache Addition durch dynamische Skalierung (Variance) und Verschiebung (Mean) der Feature Maps.
```mermaid
graph TD
    %% Inputs
    X[Rauschende Trajektorie x_t] --> MPD[MPD Layer]
    T[Zeit t] --> TE[Time Embedding]
    S[Zielform] --> SE[ShapeEncoderMPD]
    
    %% Processing Condition
    SE --> GAP((Global Average Pooling))
    GAP --> COND[1D Bedingungs-Vektor]
    
    %% Injection
    TE --> COMBINED[Kombinierter Zeit/Form Vektor]
    COND --> COMBINED
    
    COMBINED --> FILM[FiLM Projektion]
    
    %% U-Net
    MPD --> UNET[1D U-Net Backbone]
    FILM -.->|Multipliziert Gamma, addiert Beta| UNET
    
    %% Output
    UNET --> FLOW[Flow Velocity v_t]
    
    classDef bottleneck fill:#ffcccc,stroke:#ff0000,stroke-width:2px;
    class GAP bottleneck;
```

---

### 3. FiLM + Classifier-Free Guidance (`_char_film_cfg.py`)
Fügt den CFG-Dropout hinzu, um das Modell zu zwingen, stärker auf die Konditionierung zu achten, anstatt sie zu ignorieren.
```mermaid
graph TD
    %% Inputs
    X[Rauschende Trajektorie x_t] --> MPD[MPD Layer]
    T[Zeit t] --> TE[Time Embedding]
    S[Zielform] --> SE[ShapeEncoderMPD]
    
    %% Processing Condition
    SE --> GAP((Global Average Pooling))
    GAP --> COND[1D Bedingungs-Vektor]
    
    COND --> CFG{CFG Dropout}
    NULL[Lernbares Null-Token] -.->|p = 0.1| CFG
    
    %% Injection
    TE --> COMBINED[Kombinierter Zeit/Form Vektor]
    CFG --> COMBINED
    COMBINED --> FILM[FiLM Projektion]
    
    %% U-Net
    MPD --> UNET[1D U-Net Backbone]
    FILM -.->|Multipliziert Gamma, addiert Beta| UNET
    
    %% Output
    UNET --> FLOW[Flow Velocity v_t]
    
    classDef bottleneck fill:#ffcccc,stroke:#ff0000,stroke-width:2px;
    class GAP bottleneck;
    
    classDef cfg fill:#ccffcc,stroke:#00aa00,stroke-width:2px;
    class CFG cfg;
```

---

### 4. Spectral Cross-Attention (dein aktueller Run)
Ein massiver Umbau. **GAP wird komplett entfernt.** Das Spektrum wird als Sequenz von Tokens erhalten, und das U-Net "schaut" per Attention auf spezifische Frequenzen. (In deinem aktuellen Run sind CFG und die Lambda-Vorhersage deaktiviert).
```mermaid
graph TD
    %% Inputs
    X[Rauschende Trajektorie x_t] --> MPD[MPD Layer]
    T[Zeit t] --> TE[Time Embedding]
    S[Spektralkoeffizienten] --> ST[SpectralTokenizer]
    K[2D Frequenzindizes] --> ST
    
    %% Processing Condition (NO GAP!)
    ST --> KV[Keys / Values Sequenz]
    
    %% Injection (Separation of Concerns)
    TE --> FILM[FiLM Projektion]
    
    %% U-Net
    MPD --> ENC[U-Net Encoder]
    FILM -.->|NUR ZEIT!| ENC
    
    ENC --> Q[Queries: B-Spline Feature Tokens]
    Q --> BOT[U-Net Bottleneck]
    FILM -.->|NUR ZEIT!| BOT
    KV -.->|Multi-Head Cross-Attention| BOT
    
    BOT --> DEC[U-Net Decoder]
    FILM -.->|NUR ZEIT!| DEC
    KV -.->|Multi-Head Cross-Attention| DEC
    
    %% Outputs
    DEC --> FLOW[FlowHead: Flow Velocity v_t]
    
    classDef success fill:#ccddff,stroke:#0044ff,stroke-width:2px;
    class KV,Q,BOT,DEC success;
```
