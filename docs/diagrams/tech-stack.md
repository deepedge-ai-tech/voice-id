# 技术栈图

```mermaid
graph LR
    subgraph Language["语言"]
        A[Python 3.12+]
    end

    subgraph Package["包管理"]
        B[uv]
    end

    subgraph DL["深度学习"]
        C[PyTorch 2.x]
        D[torchaudio]
    end

    subgraph Voice["声纹"]
        E[pyannote.audio<br/>ResNet34]
    end

    subarray Math["数值计算"]
        F[numpy]
    end

    subgraph Test["测试"]
        G[pytest]
        H[pytest-cov]
    end

    subgraph Quality["代码质量"]
        I[black]
        J[isort]
    end

    subgraph Audio["音频增强"]
        K[audiomentations]
    end

    A --> B
    B --> C
    B --> D
    B --> E
    B --> F
    B --> G
    B --> H
    B --> I
    B --> J
    B --> K

    style Language fill:#e3f2fd
    style Package fill:#fff3e0
    style DL fill:#f3e5f5
    style Voice fill:#e8f5e9
    style Math fill:#fce4ec
    style Test fill:#e0f7fa
    style Quality fill:#f1f8e9
    style Audio fill:#fff8e1
```
