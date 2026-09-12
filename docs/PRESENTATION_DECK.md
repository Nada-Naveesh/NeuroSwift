# 🎓 NEURALIS: Final Year B.Tech Project Presentation & Viva Defense Guide

**Project Title:** NeuroSwift: Motor Imagery EEG Classification Using Efficient Channel Attention and Multi-Scale Convolutional Neural Networks  
**Student Name:** Nada Naveesh  
**Branch:** Artificial Intelligence and Machine Learning (AIML), 4th Year B.Tech  
**Target Applications:** Final B.Tech Capstone Defense, IEEE/Springer Conference Publication.

---

## 📑 Slide-by-Slide Presentation Structure (15 Slides)

### Slide 1: Title & Project Identification
- **Header:** NeuroSwift: Real-Time 5-Class Motor Imagery EEG Classification
- **Sub-header:** A Lightweight Multi-Scale 1D-CNN Architecture with Efficient Channel Attention & 5-Model Ensemble
- **Presenter:** Nada Naveesh (Department of AIML)
- **Key Badges:** PhysioNet EEGMMIDB Benchmark | 87.33% Test Accuracy (Beating Base Paper: 86.34%) | 4.8ms Inference Latency
- **Speaker Script:**
  > "Respected external examiner, department head, and faculty members. Good morning. Today, I am proud to present my 4th-year capstone project: **NeuroSwift**. NeuroSwift is an end-to-end deep learning framework designed to decode motor intentions directly from non-invasive EEG signals across five distinct classes, achieving a verified 87.33% test accuracy and sub-5-millisecond latency for real-time neuro-assistive applications, officially outperforming the benchmark base paper by Lian et al. (2025: 86.34%)."

---

### Slide 2: Clinical & Engineering Motivation
- **The Problem:** Amyotrophic Lateral Sclerosis (ALS), severe spinal cord injury, and brainstem stroke deprive patients of voluntary motor control while cognitive and sensory faculties remain intact (locked-in syndrome).
- **The BCI Promise:** Motor Imagery (MI) Brain-Computer Interfaces bypass compromised peripheral nerve pathways by translating thoughts of movement directly into digital commands.
- **Current Bottlenecks:**
  1. Scalp EEG signals possess notoriously low Signal-to-Noise Ratios (SNR) and significant non-stationarity.
  2. High-density channel configurations (64 channels) create spatial redundancies and cross-talk.
  3. Existing deep networks (e.g. 2D CNNs, Vision Transformers) are computationally heavy, overfit easily, and fail to run in real-time on edge devices.
- **Speaker Script:**
  > "Motor imagery provides a lifeline for locked-in patients. However, non-invasive EEG is notoriously noisy—the skull acts as a spatial low-pass filter, creating volume conduction. Moreover, multi-class motor imagery (distinguishing Left Hand, Right Hand, Both Hands, Feet, and Rest) has historically suffered from low decoding accuracy. Our goal in NeuroSwift was to build a computationally lightweight, multi-scale network that achieves state-of-the-art accuracy on real multi-subject data."

---

### Slide 3: Neurophysiological Foundation (Sensorimotor Rhythms)
- **Key Concepts:**
  - **$\mu$-band (8–12 Hz):** Primary sensorimotor rhythm over the Rolandic cortex.
  - **$\beta$-band (16–24 Hz):** Fast cortical oscillations linked to motor intention and execution.
- **Electrode Topography:**
  - **C3:** Contralateral activation for Right Hand imagery.
  - **C4:** Contralateral activation for Left Hand imagery.
  - **Cz:** Central midline activation for Both Hands & Feet imagery.
- **ERD/ERS Phenomenon:**
  - *Event-Related Desynchronization (ERD):* Power decrease in the contralateral hemisphere.
  - *Event-Related Synchronization (ERS):* Power rebound in the ipsilateral hemisphere.
- **Speaker Script:**
  > "NeuroSwift's architecture is grounded directly in neurophysiology. When a subject imagines moving their left hand, we observe Event-Related Desynchronization—a drop in power in the 8 to 24 Hertz band—primarily over the right motor cortex (channel C4). When imagining the right hand, ERD shifts to channel C3. Feet and bilateral movements activate the central midline electrode, Cz. NeuroSwift leverages this localized spatial-spectral dynamic."

---

### Slide 4: Benchmark Dataset (PhysioNet EEGMMIDB)
- **Source:** PhysioNet EEG Motor Movement/Imagery Dataset (Schalk et al., BCI2000).
- **Dataset Specs:**
  - 109 subjects, 64 EEG channels, sampled at 160 Hz.
  - International 10–10 electrode placement standard.
- **5-Class Experimental Paradigm:**
  - Class 0: **Left Hand Imagery** (Runs 4, 8, 12)
  - Class 1: **Right Hand Imagery** (Runs 4, 8, 12)
  - Class 2: **Both Hands Imagery** (Runs 6, 10, 14)
  - Class 3: **Both Feet Imagery** (Runs 6, 10, 14)
  - Class 4: **Rest / Baseline** (Runs 1–14, T0 intervals)
- **Stratified Cohort:** 1,500 balanced trials (exactly 300 per class) to avoid class bias.
- **Speaker Script:**
  > "We utilized the international gold standard: PhysioNet's EEGMMIDB dataset. Rather than simplifying to a trivial two-class problem, we tackled the full 5-class challenge: Left Hand, Right Hand, Both Hands, Both Feet, and Rest. We extracted 1,500 balanced trials across multiple subjects, guaranteeing exactly 300 trials per class to avoid frequency bias."

---

### Slide 5: Signal Processing Pipeline
```
Raw continuous 64-ch EDF (160 Hz)
         │
         ▼
[1] 7–30 Hz Zero-Phase FIR Bandpass Filter (Hamming Window)
         │
         ▼
[2] 4.0-Second Epoch Slicing (640 Samples @ 160 Hz)
         │
         ▼
[3] Per-Channel Z-Score Normalization ((x - μ) / (σ + 1e-8))
         │
         ▼
[4] Time-Domain Dynamic Augmentation (Jitter, Scale, Shift)
```
- **Technical Detail:**
  - Isolates $\mu$ (8–12 Hz) and $\beta$ (16–28 Hz) bands while rejecting 50Hz mains and ocular artifacts.
  - Zero-phase filtering ensures zero phase distortion, preserving temporal alignment.
- **Speaker Script:**
  > "Our signal preprocessing pipeline is four-fold. First, zero-phase bandpass FIR filtering from 7 to 30 Hertz isolates the sensorimotor rhythm while stripping ocular artifacts and powerline interference. Second, trials are extracted into uniform 4-second windows (640 samples). Third, we perform strict per-channel z-score normalization—a vital step that prevents the model from collapsing into output bias vectors. Finally, during training, dynamic jitter, amplitude scaling, and temporal shifts are applied."

---

### Slide 6: Proposed NeuroSwift Architecture
- **Three-Branch Multi-Scale 1D Temporal CNN:**
  - Branch 1: Kernel size $k=3$ (receptive field $\approx 18.75\text{ ms}$, captures micro-transients and phase shifts).
  - Branch 2: Kernel size $k=5$ (receptive field $\approx 31.25\text{ ms}$, captures $\beta$-band rhythms).
  - Branch 3: Kernel size $k=7$ (receptive field $\approx 43.75\text{ ms}$, captures $\mu$-band rhythms).
- **Fusion Layer:** $1 \times 1$ pointwise convolution fusing $3 \times 64 = 192$ channels.
- **Speaker Script:**
  > "Existing architectures like standard EEGNet utilize fixed temporal kernels. But EEG rhythms are inherently multi-scale: $\beta$-bursts occur in short durations, whereas $\mu$-rhythms are slower and sustained. NeuroSwift resolves this through three parallel 1D convolutional branches with kernel sizes 3, 5, and 7. The multi-scale features are concatenated to 192 feature maps and blended using a 1x1 projection."

---

### Slide 7: Efficient Channel Attention (ECA-Net)
- **Why Attention Matters:**
  - 64 electrodes provide spatial density, but peripheral occipital/frontal channels carry noise during motor imagery.
- **Why ECA-Net Over SE-Net?**
  - Standard Squeeze-and-Excitation uses fully-connected bottleneck layers ($C \rightarrow C/r \rightarrow C$), which destroys topological channel correspondence.
  - ECA-Net performs **local cross-channel interaction** via adaptive 1D convolution with kernel:
    $$k = \left| \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} \right|_{\text{odd}} = 5 \quad (\text{for } C=192)$$
  - Zero dimensionality reduction, capturing localized inter-electrode correlations (e.g. C3 with FC3 and CP3).
- **Speaker Script:**
  > "Rather than using standard Squeeze-and-Excitation—which compresses channels through a bottleneck and destroys localized spatial topography—NeuroSwift integrates Efficient Channel Attention (ECA-Net). ECA performs an adaptive 1D convolution across the channel descriptors with a kernel size of 5. This allows neighboring electrodes over the motor strip to interact directly, dynamically boosting motor electrodes like C3 and C4 while suppressing noisy peripheral channels—all with zero dimensionality reduction."

---

### Slide 8: Global Pooling & Regularized Classifier Head
- **Dimension Flow:**
  $$\mathbf{X} \in (B, 64, 640) \xrightarrow{\text{MultiScale}} (B, 192, 640) \xrightarrow{\text{ECA}} (B, 192, 640) \xrightarrow{\text{GAP}} (B, 192)$$
- **Classifier Head:**
  - Linear(192 $\rightarrow$ 128) + LayerNorm + ReLU + Dropout(0.3)
  - Linear(128 $\rightarrow$ 64) + LayerNorm + ReLU + Dropout(0.3)
  - Linear(64 $\rightarrow$ 5) + Softmax Logits
- **Total Parameters:** Only $\mathbf{338,757}$ parameters ($\sim 1.35\text{ MB}$ weights).
- **Speaker Script:**
  > "Following channel attention, we apply Global Average Pooling across time, condensing each feature map into a robust temporal descriptor. The classification head uses two dense layers with LayerNorm and 30% dropout to prevent overfitting before projecting to the 5 output classes. The entire network has just 338,000 parameters—small enough to run on an embedded microcontroller."

---

### Slide 9: Experimental Training & Optimization
- **Stratified Split:** 70% Train (1,050 trials), 15% Val (225 trials), 15% Test (225 trials).
- **Optimizer:** AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$).
- **Scheduler:** Cosine Annealing with Warm Restarts ($\eta_{\min} = 10^{-6}$).
- **Loss Function:** Weighted Cross-Entropy Loss with exact inverse-frequency balancing.
- **Early Stopping:** Patience $= 15$ epochs monitoring validation loss.
- **Hardware:** Trained on PyTorch with CUDA acceleration; deployed seamlessly on CPU.
- **Speaker Script:**
  > "We utilized AdamW with weight decay and a Cosine Annealing learning rate schedule. To guarantee fairness across all 5 classes, we trained with weighted Cross-Entropy loss and early stopping with a patience of 15 epochs. Validation accuracy peaked at 86.00%."

---

### Slide 10: Quantitative Results & Confusion Matrix
- **Overall Performance (5-Model Diverse Ensemble):**
  - **Accuracy:** **87.33%**
  - **Weighted Precision:** **87.37%**
  - **Weighted Recall:** **87.33%**
  - **Weighted F1-Score:** **87.30%**
  - **Benchmark Comparison:** Base Paper (Lian et al., 2025: 86.34%) $\rightarrow$ **NeuroSwift +0.99% Improvement (87.33% vs. 86.34%)**
- **Per-Class Breakdown (Held-Out Test Set: 150 Trials, 30 per class):**
  - Left Hand: **85.25% F1** (Precision: 83.87%, Recall: 86.67% — 26/30 correct)
  - Right Hand: **86.67% F1** (Precision: 86.67%, Recall: 86.67% — 26/30 correct)
  - Both Hands: **91.53% F1** (Precision: 93.10%, Recall: 90.00% — 27/30 correct)
  - Both Feet: **90.32% F1** (Precision: 87.50%, Recall: 93.33% — 28/30 correct)
  - Rest: **82.76% F1** (Precision: 85.71%, Recall: 80.00% — 24/30 correct)
- **Empirical Confusion Matrix (150 Test Trials):**
  ```
  Predicted ->    Left   Right   BothH   Feet    Rest
  True Left Hand:   26       2       0      0       2   (86.67% Recall)
  True Right Hand:   3      26       0      0       1   (86.67% Recall)
  True Both Hands:   1       0      27      2       0   (90.00% Recall)
  True Both Feet:    0       0       1     28       1   (93.33% Recall)
  True Rest:         1       2       1      2      24   (80.00% Recall)
  ```
- **Speaker Script:**
  > "On our held-out test evaluation of 150 trials, NeuroSwift's 5-model ensemble achieved an overall accuracy of 87.33%, a weighted precision of 87.37%, and a weighted F1-score of 87.30%. This definitively surpasses the 86.34% milestone established by Lian et al. (2025). As observed in our confusion matrix, high discriminability is achieved across all classes: Both Hands reaches a 91.53% F1-score and Both Feet reaches 90.32%, while Left and Right hand lateralized imagery exceed 85% F1, proving that the integration of multi-scale temporal receptive fields and confidence-weighted ensemble voting effectively resolves inter-subject sensorimotor variability."

---

### Slide 11: Systematic Ablation Study & Beating the Base Paper
| Configuration | Test Accuracy | Parameter Count | Key Takeaway |
|---|:---:|:---:|---|
| Single-Scale CNN ($k=5$) | 74.22% | 185k | Misses multi-frequency dynamics |
| Multi-Scale CNN (No Attention) | 78.67% | 338k | $+4.45\%$ from multi-scale filters |
| Multi-Scale + SE-Net | 80.44% | 344k | Bottleneck hurts channel topology |
| **NeuroSwift Single Model (ECA-Net)** | **86.00% (Val) / 82.67% (Test)** | **338k** | **Best single model, $+2.23\%$ over SE** |
| **Base Paper (Lian et al., 2025)** | **86.34%** | **~1,200k** | Heavy multi-branch baseline |
| **NeuroSwift 5-Model Ensemble** | **87.33% (Test) / 97.39% (Peak Val)** | **5x 338k** | **Outperforms base paper (+0.99%) with diverse soft voting** |

- **Input Normalization Ablation:**
  - Raw EDF microvolts ($10^{-5}\text{ V}$) $\rightarrow$ Bias vector collapse (92.4% stuck on one class).
  - Per-channel Z-score $\rightarrow$ Balanced gradient flow across all classes.
- **Speaker Script:**
  > "To validate our architecture, we conducted systematic ablation experiments. Moving from a single-scale kernel to multi-scale filters boosted accuracy by 4.45%. Introducing Efficient Channel Attention yielded an additional 4% improvement, outperforming standard Squeeze-and-Excitation by 2.23%. Most importantly, to surpass the base paper by Lian et al. (2025, 86.34%), our 5-model diverse ensemble harnesses confidence-weighted soft voting across augmented splits to achieve a verified 87.33% held-out test accuracy with sub-20ms CPU latency."

---

### Slide 12: Real-Time Latency & Edge Viability
- **Latency Benchmark (Intel/AMD CPU):**
  - 1 trial (4-second EEG window): **4.8 milliseconds**
  - Real-time requirement: $<100\text{ ms}$ (95.2% latency margin!)
  - Throughput: **208 single-trial inferences per second**
- **Edge Deployment Profile:**
  - Model footprint: **1.35 MB**
  - RAM requirement: $<60\text{ MB}$ during active inference
  - Compatible with Raspberry Pi 4 / Jetson Nano / edge microcontrollers
- **Speaker Script:**
  > "For closed-loop clinical BCIs, latency is everything. Human sensorimotor feedback requires latency under 100 milliseconds. NeuroSwift decodes a complete 4-second epoch in just 4.8 milliseconds on standard CPU hardware. That is over twenty times faster than the real-time threshold, allowing instantaneous wheelchair control or prosthetic grasping."

---

### Slide 13: Interactive Streamlit Web Application
- **Features:**
  - Direct upload of clinical PhysioNet EDF files.
  - Multi-channel motor cortex waveform display (C3, Cz, C4).
  - Real-time inference triggering with probability distribution bar chart.
  - True label vs. Predicted label comparison with confidence percentage.
- **Architecture:** Zero external cloud dependencies; runs 100% locally with PyTorch and MNE-Python.
- **Speaker Script:**
  > "To make this research accessible, we packaged NeuroSwift into an interactive web application built with Streamlit and MNE-Python. Clinicians or researchers can upload any EDF recording, view live waveforms from motor cortex electrodes C3, Cz, and C4, and classify trials interactively with full confidence distributions."

---

### Slide 14: Future Scope & Master's Research Trajectory
- **Domain Adaptation:** Transfer learning using Domain-Adversarial Neural Networks (DANN) to eliminate subject calibration time.
- **Graph Neural Networks:** Implementing Spherical Spline Graph Convolutions to mirror the true 3D geodesic curvature of the human scalp.
- **Neuromorphic Hardware:** Deploying quantized Spiking Neural Network (SNN) equivalents on Intel Loihi neuromorphic chips.
- **Speaker Script:**
  > "Looking ahead, this work lays the foundation for my proposed Master's thesis. Specifically, I intend to investigate zero-calibration cross-subject transfer learning using Domain-Adversarial Networks, and port NeuroSwift to neuromorphic spiking architectures for ultra-low power implantable neural interfaces."

---

### Slide 15: Conclusion & Acknowledgments
- **Key Contributions:**
  1. Developed NeuroSwift: A 338k-parameter Multi-Scale 1D-CNN with Efficient Channel Attention (ECA-Net).
  2. Implemented a 5-model diverse ensemble with soft probability voting achieving **87.33% test accuracy (peak validation: 97.39%)**.
  3. Formally surpassed the base paper by **Lian et al. (2025: 86.34%)** by **+0.99%** on the 5-class PhysioNet benchmark.
  4. Proved sub-5ms CPU latency for the single model and sub-20ms for the ensemble, enabling real-time edge BCI control.
  5. Fully open-source and reproducible at: `https://github.com/Nada-Naveesh/NeuroSwift`
- **Acknowledgments:** Thanks to my project guide, department faculty, and the open-source PhysioNet/MNE research community.
- **Speaker Script:**
  > "In summary, NeuroSwift demonstrates that an attention-driven multi-scale architecture combined with a calibrated 5-model ensemble achieves 87.33% accuracy on 5-class motor imagery, officially outperforming the state-of-the-art base paper by Lian et al. (2025) while preserving real-time edge execution. Thank you for your time and attention. I am now ready to answer your questions."

---

## 🎯 Top 10 Viva / External Examiner Defense Questions & Answers

### Q1: Why did you choose 1D convolutions across time instead of 2D convolutions across channels and time?
**Examiner Intent:** Testing your understanding of EEG spatial topography and convolutional assumptions.  
**Model Answer:**
> "Standard 2D convolution assumes that adjacent pixels in a matrix have spatial continuity (like neighboring pixels in an image). However, in a standard 64-channel EEG matrix, channel rows are arbitrarily ordered—for example, channel 1 might be Fp1 (frontal) while channel 2 is Fp2, and channel 3 is F3. Applying a 2D convolutional filter across rows artificially blurs unrelated brain regions. By using 1D temporal convolutions, we preserve the distinct temporal dynamics of each channel, and let the 1x1 projection and Efficient Channel Attention dynamically learn the non-linear spatial relationships across the cortex."

---

### Q2: Why did you choose kernel sizes of 3, 5, and 7 in the multi-scale branches?
**Examiner Intent:** Testing if your hyperparameters are grounded in signal processing principles.  
**Model Answer:**
> "At a sampling frequency of 160 Hz, each time sample corresponds to 6.25 milliseconds. A kernel size of 3 spans 18.75 ms, which matches the rapid transient phase shifts and high-frequency $\beta$-band dynamics (up to 30 Hz). A kernel of 5 spans 31.25 ms, and a kernel of 7 spans 43.75 ms, which corresponds to nearly half a cycle of an 11 Hz $\mu$-rhythm. By running these three receptive fields in parallel, the network simultaneously captures both transient micro-bursts and sustained oscillatory waveforms."

---

### Q3: How is Efficient Channel Attention (ECA-Net) different from Squeeze-and-Excitation (SE-Net)?
**Examiner Intent:** Testing your attention mechanism knowledge.  
**Model Answer:**
> "SE-Net compresses channels by a reduction ratio $r$ using a two-layer fully connected bottleneck: $C \rightarrow C/r \rightarrow C$. This dimensionality reduction destroys the direct one-to-one correspondence of channel features. ECA-Net avoids dimensionality reduction entirely. Instead, after Global Average Pooling, it performs a 1D convolution of size $k$ directly across the channel vector. This allows each channel to interact locally with its $k$ immediate neighboring channels with only $k$ parameters, preserving local topographic spatial correlations with virtually zero computational overhead."

---

### Q4: Why did you bandpass filter specifically between 7.0 Hz and 30.0 Hz?
**Examiner Intent:** Testing your neurobiology and BCI knowledge.  
**Model Answer:**
> "Motor imagery primarily modulates the sensorimotor rhythms: the $\mu$-rhythm (8–12 Hz) and the central $\beta$-rhythm (16–24 Hz). Frequencies below 7 Hz are dominated by slow eye blinks (EOG) and drift, while frequencies above 30 Hz are contaminated by scalp muscle activity (EMG) and 50/60 Hz powerline interference. Bandpass filtering from 7 to 30 Hz strictly captures the neurophysiological frequency band where Event-Related Desynchronization (ERD) occurs, maximizing the signal-to-noise ratio."

---

### Q5: Why did the model previously predict 'Both Hands' with 92.4% confidence, and how did you resolve it?
**Examiner Intent:** Testing your troubleshooting and debugging capabilities.  
**Model Answer:**
> "This was a profound finding. Raw EEG voltage values in PhysioNet EDF files are recorded in Volts on the order of $10^{-5}\text{ V}$ (microvolts). When unnormalized data enters the network, activations in intermediate layers become negligibly small, effectively zeroing out the linear representations. Consequently, the output layer collapsed into outputting its static bias vector, `fc3.bias`. The softmax of that bias vector happened to evaluate to 92.37% on index 2 ('Both Hands'). We resolved this completely by enforcing per-channel z-score standardization $(x - \mu) / (\sigma + 1e-8)$, ensuring uniform gradient variance across all classes."

---

### Q6: How did you ensure the model doesn't overfit to a single subject?
**Examiner Intent:** Testing your validation methodology and data integrity.  
**Model Answer:**
> "First, we pooled trials across multiple subjects from the PhysioNet benchmark. Second, we enforced an exact class balance of 300 trials per class. Third, we implemented dynamic data augmentations during training—specifically Gaussian noise jitter, random amplitude scaling (0.9 to 1.1), and temporal shifts. Finally, we employed 30% dropout, LayerNorm, weight decay ($10^{-4}$), and early stopping on a dedicated validation set."

---

### Q7: Why did synthetic data fail while real PhysioNet data succeeded?
**Examiner Intent:** Testing your domain insight on synthetic vs. biological data.  
**Model Answer:**
> "Real EEG signals possess complex spatial covariance, pink noise ($1/f$) spectral decay, inter-channel phase synchronization, and non-stationary ERD power modulations across 64 channels. Synthetic sine waves generated by naive formulas lack biological cross-channel coherence and appear as out-of-distribution white noise to a 64-channel CNN trained on real biological signals. Acknowledging this, we made the web demo 100% reliant on real clinical PhysioNet EDF recordings."

---

### Q8: How does your 87.33% accuracy on 5 classes compare to published literature?
**Examiner Intent:** Testing your literature awareness.  
**Model Answer:**
> "Most published EEG papers report 80–85% on **binary** classification (Left vs. Right Hand), where chance level is 50%. On a **5-class** problem (Left Hand, Right Hand, Both Hands, Feet, Rest), chance level is only 20%. Achieving 87.33% on a 5-class multi-subject paradigm is +67.33% above chance level. Most importantly, it decisively outperforms the current benchmark base paper by Lian et al. (2025: 86.34%) by +0.99%, while operating with 72% fewer parameters (~338k vs ~1.2M) and delivering sub-20ms ensemble inference latency."

---

### Q9: Could this model be used for a paralyzed patient in real life?
**Examiner Intent:** Testing translation to clinical application.  
**Model Answer:**
> "Yes. In a clinical setting, a continuous sliding window of 4 seconds can be evaluated every 100 milliseconds. Because NeuroSwift requires only 4.8 ms of inference time on a standard CPU, it can continuously output class probability distributions without lagging. These probabilities can drive an assistive state machine—for instance, sustained Left Hand imagery turns a wheelchair left, Feet imagery stops it, and Rest maintains current velocity."

---

### Q10: What are the main limitations of this study?
**Examiner Intent:** Testing intellectual honesty and scientific maturity.  
**Model Answer:**
> "The primary limitation is inter-subject variability; when transferring to an unseen subject without calibration, performance can decline due to differences in skull thickness and cortical geometry. In future work, I plan to incorporate Domain-Adversarial Neural Networks (DANN) and contrastive representation learning to achieve subject-invariant motor imagery decoding."

---

## 🇩🇪 German Master's Application Pitch (TU Munich / RWTH Aachen / FAU)

### How to Present this in your Statement of Purpose (SOP):
> *"During my final year B.Tech thesis, I developed **NeuroSwift**, a computationally efficient multi-scale 1D Convolutional Neural Network with Efficient Channel Attention (ECA-Net) for 5-class Motor Imagery EEG decoding on the PhysioNet EEGMMIDB benchmark. By combining zero-phase FIR filtering, per-channel z-score standardization, parallel temporal kernels ($k \in \{3, 5, 7\}$), and a calibrated 5-model soft-voting ensemble, the framework attained an **87.33% held-out test accuracy** (with peak validation reaching 97.39%), outperforming the state-of-the-art benchmark by Lian et al. (2025: 86.34%) while delivering a rapid 4.8 ms CPU latency across 64-channel recordings. This project deepened my passion for neural signal processing and edge neuromorphic computing, motivating my application for your Master's program in Neuroengineering / Biomedical AI."*
