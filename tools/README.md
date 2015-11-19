Tools for training, testing, and compressing Fast R-CNN networks.

origin:
	Train
		cd tools/
		// CaffeNet
		python train_net.py --gpu 0 --solver ../models/CaffeNet/solver.prototxt \
				--weights ../data/imagenet_models/CaffeNet.v2.caffemodel
		// VGGNet 
		python train_net.py --gpu 0 --solver ../models/VGG16/solver.prototxt \
				--weights ../data/imagenet_models/VGG16.v2.caffemodel

	Test:
		cd tools/
		// CaffeNet
		python test_net.py --gpu 0 --def ../models/CaffeNet/test.prototxt \
				--net ../output/default/voc_2007_trainval/caffenet_fast_rcnn_iter_40000.caffemodel
		// VGGNet
		python test_net.py --gpu 0 --def ../models/VGG16/test.prototxt \
				--net ../output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel

	Compress
		cd tools/
		// compress
		python compress_net.py --def models/VGG16/test.prototxt \
				--def-svd models/VGG16/compressed/test.prototxt \
				--net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000.caffemodel

		// Test the model you just compressed
		python test_net.py --gpu 0 --def models/VGG16/compressed/test.prototxt \
    		--net output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_40000_svd_fc6_1024_fc7_256.caffemodel		