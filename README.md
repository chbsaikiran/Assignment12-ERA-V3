---
title: NextWordGPT
emoji: 🏃
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
short_description: 'Transformer trained on Shakespearean text '
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


<pre>
Epoch 1/50: 100%|██████████| 82/82 [01:16<00:00,  1.08step/s, loss=6.2489]
Epoch 1/50, Loss: 7.0745, Time: 76.07s
Epoch 2/50: 100%|██████████| 82/82 [01:22<00:00,  1.00s/step, loss=5.6592]
Epoch 2/50, Loss: 5.6716, Time: 82.14s
Epoch 3/50: 100%|██████████| 82/82 [01:25<00:00,  1.05s/step, loss=5.2294]
Epoch 3/50, Loss: 5.1465, Time: 85.97s
Epoch 4/50: 100%|██████████| 82/82 [01:27<00:00,  1.07s/step, loss=4.8800]
Epoch 4/50, Loss: 4.8121, Time: 87.40s
Epoch 5/50: 100%|██████████| 82/82 [01:28<00:00,  1.08s/step, loss=4.6155]
Epoch 5/50, Loss: 4.5597, Time: 88.28s
Epoch 6/50: 100%|██████████| 82/82 [01:29<00:00,  1.10s/step, loss=4.4006]
Epoch 6/50, Loss: 4.3344, Time: 89.88s
Epoch 7/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=4.1696]
Epoch 7/50, Loss: 4.1084, Time: 91.19s
Epoch 8/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=3.9078]
Epoch 8/50, Loss: 3.8753, Time: 91.43s
Epoch 9/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=3.6197]
Epoch 9/50, Loss: 3.6167, Time: 91.38s
Epoch 10/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=3.3067]
Epoch 10/50, Loss: 3.3436, Time: 91.24s
Epoch 11/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=3.0890]
Epoch 11/50, Loss: 2.9951, Time: 91.45s
Epoch 12/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=2.7631]
Epoch 12/50, Loss: 2.7189, Time: 91.25s
Epoch 13/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=2.5140]
Epoch 13/50, Loss: 2.4935, Time: 91.21s
Epoch 14/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=2.3475]
Epoch 14/50, Loss: 2.3095, Time: 91.42s
Epoch 15/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=2.1527]
Epoch 15/50, Loss: 2.1343, Time: 91.61s
Epoch 16/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=1.9820]
Epoch 16/50, Loss: 1.9522, Time: 91.35s
Epoch 17/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=1.7411]
Epoch 17/50, Loss: 1.7585, Time: 91.53s
Epoch 18/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=1.5516]
Epoch 18/50, Loss: 1.5744, Time: 91.77s
Epoch 19/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=1.3633]
Epoch 19/50, Loss: 1.4087, Time: 91.45s
Epoch 20/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=1.2165]
Epoch 20/50, Loss: 1.2397, Time: 91.37s
Epoch 21/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=1.1129]
Epoch 21/50, Loss: 1.0790, Time: 91.69s
Epoch 22/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.9431]
Epoch 22/50, Loss: 0.9302, Time: 91.61s
Epoch 23/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=0.8262]
Epoch 23/50, Loss: 0.8121, Time: 91.39s
Epoch 24/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=0.7406]
Epoch 24/50, Loss: 0.7170, Time: 91.36s
Epoch 25/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.6618]
Epoch 25/50, Loss: 0.6387, Time: 91.58s
Epoch 26/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.5878]
Epoch 26/50, Loss: 0.5709, Time: 91.55s
Epoch 27/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=0.5246]
Epoch 27/50, Loss: 0.5079, Time: 91.23s
Epoch 28/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=0.4453]
Epoch 28/50, Loss: 0.4472, Time: 91.39s
Epoch 29/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.3966]
Epoch 29/50, Loss: 0.3912, Time: 91.58s
Epoch 30/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=0.3454]
Epoch 30/50, Loss: 0.3401, Time: 91.14s
Epoch 31/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=0.3288]
Epoch 31/50, Loss: 0.3059, Time: 91.06s
Epoch 32/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=0.2900]
Epoch 32/50, Loss: 0.2712, Time: 91.22s
Epoch 33/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.2608]
Epoch 33/50, Loss: 0.2438, Time: 91.44s
Epoch 34/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=0.2365]
Epoch 34/50, Loss: 0.2215, Time: 91.02s
Epoch 35/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=0.2159]
Epoch 35/50, Loss: 0.2017, Time: 91.14s
Epoch 36/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.1979]
Epoch 36/50, Loss: 0.1840, Time: 91.59s
Epoch 37/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.1814]
Epoch 37/50, Loss: 0.1681, Time: 91.70s
Epoch 38/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.1661]
Epoch 38/50, Loss: 0.1539, Time: 91.46s
Epoch 39/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.1522]
Epoch 39/50, Loss: 0.1410, Time: 91.53s
Epoch 40/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.1390]
Epoch 40/50, Loss: 0.1295, Time: 91.60s
Epoch 41/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.1350]
Epoch 41/50, Loss: 0.1215, Time: 91.51s
Epoch 42/50: 100%|██████████| 82/82 [01:31<00:00,  1.11s/step, loss=0.1304]
Epoch 42/50, Loss: 0.1156, Time: 91.43s
Epoch 43/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.1247]
Epoch 43/50, Loss: 0.1099, Time: 91.80s
Epoch 44/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.1162]
Epoch 44/50, Loss: 0.1047, Time: 91.56s
Epoch 45/50: 100%|██████████| 82/82 [01:31<00:00,  1.12s/step, loss=0.1122]
Epoch 45/50, Loss: 0.0998, Time: 91.53s
</pre>
