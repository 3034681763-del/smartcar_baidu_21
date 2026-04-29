# SmartCar Baidu 21 - 智慧农场智能车

面向第二十一届全国大学生智能汽车竞赛百度智慧交通创意组“智慧农场”赛题的 Jetson 上位机工程。项目运行在 Jetson Orin Nano 上，负责巡线、目标检测、OCR、大模型推理、任务状态机调度，并通过串口与下位机/机械臂协同完成比赛任务。

## 项目特点

- Jetson Orin Nano 上位机主控
- Paddle Lite 巡线模型 + 目标检测模型
- 任务摄像头与巡线摄像头双路输入
- OCRPipeline 文本识别链路
- AI Studio / 千帆双后端大模型调用
- 串口队列式底盘、机械臂、系统模式指令发送
- 基于 `world_y`、航向角、目标检测结果的任务状态机
- 支持新旧模型路径一键切换

## 当前任务链路

主流程由 `main.py` 启动，状态机在 `State_Handle.py` 中调度，任务执行代码主要在 `move_base.py` 和各任务文件中。

```text
Lane 巡线
-> Seeding 寻址播种
-> PestConfirm 有害/有益动物确认
-> Irrigation 智能灌溉
-> Shooting 农田除害射击
-> Harvest 作物采收
-> Sort 分拣入库
-> OrderDelivery 智能接单与取货
-> PlaceDelivery 产品配送
-> Lane / Stop
```

## 目录结构

```text
main.py                  主程序入口
State_Handle.py          状态机与任务触发
move_base.py             底盘/机械臂指令封装与任务聚合
SerialCommunicate.py     串口通信服务
Cam_cap.py               双摄像头采集与巡线模型推理
infer_server_client.py   目标检测/OCR 模型服务与客户端
motion.json              底盘、机械臂动作组和任务配置
param.json               PID 等参数配置

harvest_task.py          任务4 作物采收
sort_task.py             任务5 分拣入库
order_delivery_task.py   任务6 智能接单与取货
place_delivery_task.py   任务7 产品配送
shooting_task.py         农田除害射击
pest_confirm_task.py     有害/有益动物确认
irrigation_task.py       智能灌溉

order_ai_client.py       订单文本大模型解析
pest_vlm_client.py       有害动物多模态大模型识别
llm_api_test.py          大模型全链路测试脚本
model_path_config.py     新旧模型路径切换配置
env_loader.py            本地 .env 配置加载
```

## Jetson 快速启动

拉取代码：

```bash
cd ~
git clone https://github.com/3034681763-del/smartcar_baidu_21.git
cd smartcar_baidu_21
```

如果仓库已经存在：

```bash
cd ~/smartcar_baidu_21
git pull --rebase origin main
```

启动主程序：

```bash
python main.py
```

## 模型路径切换

默认使用旧路径：

```text
巡线模型: /home/jetson/workspace_plus/vehicle_wbt_21th_lane/src/cnn_auto.nb
目标检测: Global_V2
```

如果 Jetson 仓库下存在新的 `src` 目录：

```text
src/cnn_lane.nb
src/target_det/
```

运行前输入：

```bash
export SMARTCAR_MODEL_PROFILE=src
python main.py
```

切回旧路径：

```bash
export SMARTCAR_MODEL_PROFILE=legacy
python main.py
```

也可以写入本地 `.env`，之后无需每次 `export`：

```bash
echo "SMARTCAR_MODEL_PROFILE=src" >> .env
```

单独覆盖某个模型：

```bash
export LANE_MODEL_PATH=/home/jetson/smartcar_baidu_21/src/cnn_lane.nb
export TASK_MODEL_PATH=/home/jetson/smartcar_baidu_21/src/target_det
python main.py
```

## 大模型 API 配置

默认使用 PPT 中展示的 AI Studio OpenAI-compatible API。

创建本地 `.env`：

```bash
cp .env.example .env
nano .env
```

推荐配置：

```bash
LLM_PROVIDER=aistudio
AI_STUDIO_API_KEY=你的_AiStudio_Access_Token
AI_STUDIO_TEXT_MODEL=ernie-x1-turbo-32k
AI_STUDIO_VLM_MODEL=ernie-4.5-vl-28b-a3b
SMARTCAR_MODEL_PROFILE=src
```

安装接口依赖：

```bash
pip install openai
```

如需切回千帆：

```bash
export LLM_PROVIDER=qianfan
export QIANFAN_API_KEY=你的千帆Key
```

`.env` 已被 `.gitignore` 忽略，不要把真实密钥提交到仓库。

## 常用测试命令

摄像头测试：

```bash
python camera_test.py
```

目标检测测试：

```bash
export SMARTCAR_MODEL_PROFILE=src
python infer_test.py --device 1-2.1:1.0
```

底盘动作测试：

```bash
python base_motion_test.py --actions moveshort backshort
```

复杂底盘动作：

```bash
python base_motion_test.py --actions moveshort TurnLeft TurnLeft backshort TurnRight TurnRight
```

串口测试：

```bash
python serial_test.py
```

大模型订单全链路测试：

```bash
python llm_api_test.py --skip-pest
```

大模型有害动物多模态测试：

```bash
python llm_api_test.py --skip-order
```

`llm_api_test.py` 会调用任务摄像头，预览画面后按 `q` 抓拍，然后继续走检测、OCR、裁剪和大模型 API。

## 串口与 ACK

底盘动作组通过 `motion.json` 中的 `BASE_MOTION` 发送：

```json
{"cmd": "Motion", "mode": 1, "pos_x": 0, "pos_y": 60, "z_angle": 0}
```

下位机动作完成后需要返回 ACK：

```text
42 05 01 05 3C
```

当前代码中底盘动作会连续发送 10 遍，以降低串口丢包影响，然后等待下位机 ACK。

## 开发流程建议

本地电脑修改代码：

```bash
git add 文件名
git commit -m "说明本次改动"
git push origin main
```

Jetson 拉取：

```bash
cd ~/smartcar_baidu_21
git pull --rebase origin main
```

如果 Jetson 本地有临时改动导致冲突，优先查看：

```bash
git status -sb
git diff -- 文件名
```

确认临时改动不需要后：

```bash
git restore 文件名
git pull --rebase origin main
```

## 注意事项

- 不要使用 `git add .` 直接提交本地模型、日志或密钥文件。
- `src/` 模型文件建议只保存在 Jetson 本地，除非确认要纳入仓库。
- `.env` 存放本地密钥和路径开关，不应提交。
- 如果大模型接口报 401/403，检查 AI Studio Access Token 和模型权限。
- 如果底盘动作卡住，优先检查下位机是否返回 ACK。
- 如果切换模型后检测异常，先用 `infer_test.py` 单独验证目标检测模型。

## 当前重点 TODO

- 继续实车标定 `motion.json` 中机械臂动作组
- 完善任务 6/7 的 OCR 稳定性与异常重试
- 实测 AI Studio 多模态模型在动物卡片上的判断稳定性
- 根据赛场实际光照调整目标检测阈值与相机参数
