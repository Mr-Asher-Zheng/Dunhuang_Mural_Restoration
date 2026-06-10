# Dunhuang_Mural_Restoration
敦煌壁画修复

# 代码
```python
# 计算不同频率的指数衰减
embeddings = math.log(10000) / (half_dim - 1)
# 生成频率序列
embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
```

<img width="966" height="341" alt="image" src="https://github.com/user-attachments/assets/ca89fdab-092a-4703-a8f9-1c5ec8805f43" />
<img width="970" height="536" alt="image" src="https://github.com/user-attachments/assets/a8b1f3bc-9ee6-4e59-a345-a808c109801e" />
<img width="966" height="566" alt="image" src="https://github.com/user-attachments/assets/f0c036f8-a2c3-4d5c-a718-c57d6b1c0894" />

```python
# 将时间步与频率序列相乘
# 对应t⋅freq_i
embeddings = time[:, None] * embeddings[None, :]
```

```python
# 拼接sin和cos得到最终的嵌入向量
embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
```
