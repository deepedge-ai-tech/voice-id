# 时序图 (Sequence Diagram)

## WespeakerDeep 注册流程时序图（当前最优）

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant D as WespeakerDeep
    participant C as WespeakerClient
    participant L as AudioLoader
    participant M as ResNet34模型
    participant S as 存储(.pkl)

    U->>D: enroll(clean_dir, pk_path)
    D->>D: 初始化 DeepConfig
    D->>C: _ensure_model()
    C-->>D: model ready

    loop 每个 .wav 文件 in clean_dir
        D->>L: _load_audio(file, 16000)
        L-->>D: waveform (16kHz, mono)

        Note over D: 跳过 VAD (enroll_skip_vad=True)

        D->>M: _extract_embedding(waveform)
        M-->>D: 256 维 embedding

        D->>D: F.normalize(dim=0)
        D->>D: 收集到 all_embeddings[]
    end

    D->>D: reference = mean(templates) + normalize
    D->>D: 构建 dict{version, templates[], reference}
    D->>S: pickle.dump(dict)
    S-->>D: 保存成功
    D-->>U: {ok, num_segments, num_templates, pk_path}
```

## WespeakerDeep 识别流程时序图（当前最优）

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant D as WespeakerDeep
    participant S as 存储(.pkl)
    participant L as AudioLoader
    participant M as ResNet34模型

    U->>D: recognize(audio.wav, voice.pkl)
    D->>D: _ensure_model()

    D->>S: load_full(pk_path) — 所有模板
    S-->>D: {version, templates[], reference}

    D->>L: _load_audio(path, 16000)
    L-->>D: waveform

    D->>D: head_window 裁剪
    Note over D: 超长音频保留头部 60s

    alt 短音频 (< 1.5s) 且启用滑动窗口
        D->>D: 滑动窗口 0.4s/0.15s
        D->>M: 每个窗口提取 embedding
        M-->>D: N 个 window embedding
        D->>D: 多模板匹配: max(cosine(w, t) for w, t)
    else 长音频
        D->>M: 单次 _extract_embedding
        M-->>D: test_embedding
        D->>D: 多模板匹配: max(cosine(test, t) for t in templates)
    end

    D->>D: sqrt 分数补偿
    Note over D: factor = min((2.0/dur)^0.5, 2.0)
    D->>D: 比较 compensated >= 0.50

    D-->>U: {is_recognized, confidence, raw_confidence, threshold, num_templates_used, sliding_windows_used, score_compensation_factor}
```

---

## 旧版 WespeakerClient 流程（Legacy）

### 注册流程时序图

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant C as WespeakerClient
    participant L as AudioLoader
    participant V as SileroVAD
    participant A as NoiseAugmentor
    participant M as ResNet34模型
    participant S as 存储(.pkl)

    U->>C: enroll(audio.wav, voice.pkl)
    C->>C: _ensure_model()
    C->>L: _load_audio(path, 16000)
    L-->>C: waveform (16kHz, mono)

    C->>C: 切分为 1s 片段

    opt 启用噪声增强
        C->>A: augment(segments)
        A-->>C: 增强后片段
    end

    loop 每个片段
        C->>M: _extract_embedding(waveform)
        M-->>C: 256 维 embedding
    end

    opt 启用 SNR 加权
        C->>C: _estimate_snr() 计算权重
        C->>C: 加权平均 + 归一化
    else 普通平均
        C->>C: 均值 + 归一化
    end

    C->>S: pickle.dump(embedding)
    S-->>C: 保存成功
    C-->>U: {ok: true, num_segments: N, pk_path}
```

### 识别流程时序图

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant C as WespeakerClient
    participant S as 存储(.pkl)
    participant L as AudioLoader
    participant V as SileroVAD
    participant M as ResNet34模型

    U->>C: recognize(audio.wav, voice.pkl)
    C->>C: _ensure_model()

    C->>S: 加载参考声纹
    S-->>C: reference_embedding

    C->>L: _load_audio(path, 16000)
    L-->>C: waveform

    C->>C: _crop_verify() 裁剪窗口

    opt 启用 VAD
        C->>V: _vad_segments(waveform)
        V-->>C: 语音片段列表
    end

    C->>M: _extract_embedding(processed_audio)
    M-->>C: test_embedding

    C->>C: 归一化 + 余弦相似度
    C->>C: 与阈值比较

    C-->>U: {is_recognized, confidence, threshold}
```
