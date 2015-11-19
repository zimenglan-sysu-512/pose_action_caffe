# proto_file="../../models/Pose/FLIC/zhangll/lecun0-8x-3b/train_val.prototxt"
# image_file="../../models/Pose/FLIC/zhangll/lecun0-8x-3b/tb_train_val.jpg"

# proto_file="../../models/Pose/FLIC/zhangll/lecun0-8x-3b-addc4_2/train_val.prototxt"
# image_file="../../models/Pose/FLIC/zhangll/lecun0-8x-3b-addc4_2/tb_train_val.jpg"

# proto_file="../../models/Pose/FLIC/liangzj/lecun0-8x-2b-addc4_2/train_val.prototxt"
# image_file="../../models/Pose/FLIC/liangzj/lecun0-8x-2b-addc4_2/tb_train_val.jpg"

proto_file="../../models/Pose/FLIC/liangzj/lecun0-8x-2b/train_val.prototxt"
image_file="../../models/Pose/FLIC/liangzj/lecun0-8x-2b/tb_train_val.jpg"

# proto_file="../../models/Pose/FLIC/ddk/lecun0-8x/train_val.prototxt"
# image_file="../../models/Pose/FLIC/ddk/lecun0-8x/tb_train_val.jpg"

# proto_file="../../models/Pose/FLIC/ddk/lecun0-8x-addc4_2/train_val.prototxt"
# image_file="../../models/Pose/FLIC/ddk/lecun0-8x-addc4_2/tb_train_val.jpg"

rankdir="TB"
python draw_net.py $proto_file $image_file --rankdir=$rankdir


# python draw_net.py --input_net_proto_file="../../models/Pose/FLIC/zhangll3/lecun0-8x-3b/train_val.prototxt" --output_image_file="../../models/Pose/FLIC/zhangll3/lecun0-8x-3b/train_val.jpg"