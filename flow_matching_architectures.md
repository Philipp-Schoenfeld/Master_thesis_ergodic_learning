# Architektur-Visualisierungen der Flow-Matching Modelle

Dieses Dokument enthält diagrammatische Visualisierungen der verschiedenen Flow-Matching Architekturen im Projekt. Jede Architektur wurde für unterschiedliche Arten der Konditionierung entwickelt.

Die Entwicklung zeigt einen klaren Trend:
1. **Frühere Ansätze (`_char_film_cfg.py`)**: Nutzen Global Average Pooling (MPD) der Ziel-Shapes und injizieren dies zusammen mit der Zeit über **FiLM** (Feature-wise Linear Modulation) in alle CNN-Blöcke.
2. **Neuere Ansätze (`_spectral`, `_particles`, `_waypoint`)**: Trennen die Zeit- und Form-Konditionierung. Die Zeit wird weiterhin über FiLM injiziert, aber die geometrische/spektrale Konditionierung erfolgt viel ausdrucksstärker über **Cross-Attention** in den Bottleneck- und Decoder-Schichten des U-Nets.
3. **Training mit direkten Metriken**: Neben dem reinen Flow-Matching (CFM) MSE Loss gibt es hybride Ansätze (CFM + Ergodic Loss auf Endpoint-Schätzung) und reine Self-Supervised Ansätze, die direkt gegen die Ergodische Energie optimieren und Flow Matching komplett umgehen.

---

## 1. Baseline: Conditional MPD U-Net (Shape Context via FiLM)
**Datei:** `flow_matching_cond_mpd_unet_char_film_cfg.py`

Diese Architektur bündelt die Zielform in einen einzigen globalen Kontextvektor (mittels Mean Pooling) und injiziert diesen Vektor zusammen mit der Zeit in das gesamte Netzwerk.

```mermaid
graph TD
    subgraph Inputs
        X["Trajektorie x_t (B, nxi, nd)"]
        T["Zeit t (B,)"]
        REF["Referenz-Shape (B, nxi, nd)"]
    end

    subgraph Tokenization
        MPD_X["MPDLayer + PosEmb"]
        TimeEmb["SinusoidalTimeEmbedding"]
        ShapeEnc["ShapeEncoderMPD (MPDLayer + MeanPool + MLP)"]
    end
    
    X --> MPD_X
    T --> TimeEmb
    REF --> ShapeEnc

    subgraph Conditioning
        CFG1["CFG Dropout"]
        CondAdd["Add (Zeit-Emb + Shape-Kontext)"]
    end
    
    ShapeEnc --> CFG1
    CFG1 --> CondAdd
    TimeEmb --> CondAdd
    
    subgraph Backbone
        UNET["UNetBackbone (1D CNN)"]
        FiLM["FiLM Konditionierung"]
    end

    MPD_X --> UNET
    CondAdd --> FiLM
    FiLM -. "skaliert verschiebt Feature Maps" .-> UNET

    subgraph Output
        Head["OutputMLPHead"]
        OutV["Geschwindigkeit v_t (B, nxi, nd)"]
    end

    UNET --> Head
    Head --> OutV
```

---

## 2. Spectral Cross-Attention
**Datei:** `flow_matching_cond_spectral_crossattn.py`

Diese Architektur nutzt Cross-Attention, um das Modell selektiv auf verschiedene Ergodische Frequenzen (Spektral-Koeffizienten) achten zu lassen. Zudem sagt sie Lagrangemultiplikatoren voraus.

```mermaid
graph TD
    subgraph Inputs
        X["Trajektorie x_t (B, nxi, nd)"]
        T["Zeit t (B,)"]
        SPEC["Spektral-Koeffizienten (B, S, 2)"]
    end

    subgraph Tokenization
        MPD_X["MPDLayer + PosEmb"]
        TimeEmb["SinusoidalTimeEmbedding"]
        SpecTok["SpectralTokenizer (MLP pro Freq + 2D PosEnc)"]
    end
    
    X --> MPD_X
    T --> TimeEmb
    SPEC --> SpecTok

    CFG["CFG Dropout"]
    SpecTok --> CFG

    subgraph Backbone
        UNET["UNetBackboneSpectral (1D CNN + Self-Attention)"]
        FiLM["FiLM (Nur Zeit)"]
        CrossAttn["Cross-Attention (Spektral-Tokens)"]
    end

    MPD_X --> UNET
    TimeEmb --> FiLM
    FiLM -. "zeitliches Bewusstsein" .-> UNET
    CFG --> CrossAttn
    CrossAttn -. "Fokus auf Frequenzen" .-> UNET

    subgraph Outputs
        FlowH["FlowHead"]
        LamH["LambdaHead (MeanPool + MLP)"]
        OutV["Geschwindigkeit v_t"]
        OutLam["Lagrangemultiplikatoren lambda_0"]
    end

    UNET --> FlowH
    UNET --> LamH
    FlowH --> OutV
    LamH --> OutLam
```

---

## 3. Particle Cross-Attention (Standard Training)
**Datei:** `flow_matching_cond_particles_crossattn.py`

Diese Architektur verwendet eine dichte Punktwolke (Partikel mit Dichte mu) als Konditionierung. Sie verwendet Gaussian Fourier Features, um Spectral Bias bei den geometrischen Koordinaten zu verhindern.

```mermaid
graph TD
    subgraph Inputs
        X["Trajektorie x_t (B, nxi, nd)"]
        T["Zeit t (B,)"]
        PART["Partikel x, y, mu (B, N, 3)"]
    end

    subgraph Tokenization
        MPD_X["MPDLayer + PosEmb"]
        TimeEmb["SinusoidalTimeEmbedding"]
        PartTok["ParticleTokenizer (Gaussian Fourier Feat + MLP)"]
    end
    
    X --> MPD_X
    T --> TimeEmb
    PART --> PartTok

    CFG["CFG Dropout"]
    PartTok --> CFG

    subgraph Backbone
        UNET["UNetBackboneParticles (1D CNN + Self-Attention)"]
        FiLM["FiLM (Nur Zeit)"]
        CrossAttn["Cross-Attention (Partikel-Tokens)"]
    end

    MPD_X --> UNET
    TimeEmb --> FiLM
    FiLM -. "zeitliches Bewusstsein" .-> UNET
    CFG --> CrossAttn
    CrossAttn -. "Fokus auf dichte Regionen" .-> UNET

    subgraph Output
        FlowH["FlowHead"]
        OutV["Geschwindigkeit v_t"]
    end

    UNET --> FlowH
    FlowH --> OutV
```

---

## 4. Waypoint Cross-Attention
**Datei:** `flow_matching_cond_waypoint_crossattn.py`

Hier werden Wegpunkte direkt als Tokens verarbeitet (durch 1D Convolutions auf den Koordinaten), um lokale geometrische Zusammenhänge zu lernen. Auch hier werden Lagrangemultiplikatoren vorhergesagt.

```mermaid
graph TD
    subgraph Inputs
        X["Trajektorie x_t (B, nxi, nd)"]
        T["Zeit t (B,)"]
        WAY["Wegpunkte (B, nxi, 2)"]
    end

    subgraph Tokenization
        MPD_X["MPDLayer + PosEmb"]
        TimeEmb["SinusoidalTimeEmbedding"]
        WayTok["WaypointTokenizer (1D Conv + MLP)"]
    end
    
    X --> MPD_X
    T --> TimeEmb
    WAY --> WayTok

    CFG["CFG Dropout"]
    WayTok --> CFG

    subgraph Backbone
        UNET["UNetBackboneWaypoint (1D CNN + Self-Attention)"]
        FiLM["FiLM (Nur Zeit)"]
        CrossAttn["Cross-Attention (Wegpunkt-Tokens)"]
    end

    MPD_X --> UNET
    TimeEmb --> FiLM
    FiLM -. "zeitliches Bewusstsein" .-> UNET
    CFG --> CrossAttn
    CrossAttn -. "Fokus auf lokale Geometrie" .-> UNET

    subgraph Outputs
        FlowH["FlowHead"]
        LamH["LambdaHead"]
        OutV["Geschwindigkeit v_t"]
        OutLam["Lagrangemultiplikatoren lambda_0"]
    end

    UNET --> FlowH
    UNET --> LamH
    FlowH --> OutV
    LamH --> OutLam
```

---

## 5. Flow-Matching mit Ergodischer Metrik im Loss
**Datei:** `flow_matching_cond_particles_crossattn.py` (`compute_particle_cfm_loss`)

Hierbei handelt es sich um eine abweichende Trainings- und Loss-Strategie, nicht um eine modifizierte Modell-Architektur. Das Modell ist das `ParticleCrossAttnFlowNetwork`. Die vorhergesagte Geschwindigkeit v_t wird verwendet, um den Endpunkt der Trajektorie x_1 zu schätzen. Auf diese Schätzung wird die Ergodische Metrik angewandt.

```mermaid
graph TD
    subgraph Inputs
        X1["Ziel-Trajektorie x_1"]
        X0["Rauschen x_0"]
        PART["Partikel x, y, mu (B, N, 3)"]
        T["Zeit t"]
    end

    subgraph Forward Pass CFM
        Interpolate["xt = (1-t)x0 + t*x1"]
        Net["ParticleCrossAttnFlowNetwork"]
        Interpolate --> Net
        PART --> Net
        T --> Net
        Net --> Vt["Geschwindigkeit v_t"]
    end
    
    subgraph Loss Computation
        Ut["Wahre Geschw. u_t = x1 - x0"]
        CFMLoss["CFM Loss: MSE(v_t, u_t)"]
        Vt --> CFMLoss
        Ut --> CFMLoss
        
        X1_Hat["Endpoint-Schaetzung: x1_hat = xt + (1-t)v_t"]
        Vt --> X1_Hat
        Interpolate --> X1_Hat
        
        ErgLoss["Ergodic Energy Loss (x1_hat vs. Partikel)"]
        X1_Hat --> ErgLoss
        PART --> ErgLoss
        
        TotalLoss["Gesamt = CFM_Loss + w * Erg_Loss"]
        CFMLoss --> TotalLoss
        ErgLoss --> TotalLoss
    end
```

---

## 6. Self-Supervised Particle Generator
**Datei:** `flow_matching_particles_selfsupervised.py`

Dieser Ansatz bricht mit dem iterativen Flow-Matching Konzept und nutzt das modifizierte `UNetBackboneParticles` als Single-Pass Generator. Rauschen (z) nimmt die Rolle der initialen Sequenz ein. Das Netzwerk hat keine Zeit-Eingabe (FiLM ist inaktiviert) und kein CFG (Classifier-Free Guidance). Das Output-Modell projiziert (Rauschen + Partikel-Kontext) direkt auf generierte B-Spline Kontrollpunkte.

```mermaid
graph TD
    subgraph Inputs
        Z["Rauschen z (B, nxi, nd)"]
        PART["Partikel x, y, mu (B, N, 3)"]
    end

    subgraph Tokenization
        MPD_Z["MPDLayer + PosEmb"]
        PartTok["ParticleTokenizer"]
    end
    
    Z --> MPD_Z
    PART --> PartTok

    subgraph Backbone
        UNET["UNetBackboneParticles (1D CNN + Self-Attention)"]
        CrossAttn["Cross-Attention (Partikel)"]
    end

    MPD_Z --> UNET
    PartTok --> CrossAttn
    CrossAttn -. "Fokus auf dichte Regionen" .-> UNET
    
    Note["(Kein Zeit-Emb, kein FiLM, kein CFG)"]

    subgraph Output
        Head["FlowHead"]
        Shift["+ 0.5 Offset"]
        OutXi["Generierte Trajektorie xi (B, nxi, nd)"]
    end

    UNET --> Head
    Head --> Shift
    Shift --> OutXi

    subgraph Loss Self-Supervised
        Energy["Ergodische Energie (E)"]
        Div["Diversitaets-Belohnung"]
        Total["Loss = mean(E) - w * Diversitaet"]
    end
    
    OutXi --> Energy
    OutXi --> Div
    Energy --> Total
    Div --> Total
```
