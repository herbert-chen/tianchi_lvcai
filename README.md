# 天池图像比赛Baseline分享
[2018广东工业智造大数据创新大赛——智能算法赛](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165320.5678.1.54114443WSKVPP&raceId=231682)，未调参情况下线上`0.921`
---
## 运行代码前，需要将图片放在data目录下，目录树如下：

	|--data
		|--guangdong_round1_train1_20180903
		|--guangdong_round1_train2_20180916
		|--guangdong_round1_test_a_20180916
	|--gen_label_csv.py
	|--model_v4.py
	|--main_inception_v4.py

---
## 代码运行方式：
	python gen_label_csv.py
	python main_inception_v4.py

---
## 程序说明：
框架：Pytorch 0.4

代码经测试，线上分数为`0.921`，线上19名（截至2018.9.20）。

如果有版本迭代，只需修改主程序名，则代码会自动生成新目录用于保存模型与结果文件。

使用的模型为ImageNet预训练的inception v4，整体思路：数据增强之后直接将网络全部层一起训练，做简单分类。

本代码为baseline，未经任何调参，训练参数和数据增强方式等可以自行随意修改。

输入尺寸为384×384，如果出现out of memory（显存不足），可减小batch size再重新跑。

注：只用到了guangdong_round1_train2_20180916数据，没有使用guangdong_round1_train1_20180903，可自行添加。

    许多同学反应Windows下训练时train_loader报错的问题，将workers设为0即可解决。

## 复赛方案
https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.9.743c2881MOwYf2&postId=38469
