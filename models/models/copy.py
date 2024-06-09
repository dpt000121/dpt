# 打开原始文件和目标文件
with open('/home/slr/mount/code/WHUsarasnet/testjiu.txt', 'r') as source_file, open('/home/slr/mount/code/WHUsarasnet/test.txt', 'w') as target_file:
    # 逐行读取原始文件内容
    for line in source_file:
        # 去除行尾的换行符，并添加一个制表符和相同的内容
        # line = "train_dataset/"+"train_dataset/"+"image1/"+line.strip() + ' ' + "train_dataset/"+"train_dataset/"+"image2/"+line.strip() + ' ' + "train_dataset/"+"train_dataset/"+"gt/"+line.strip() + '\n'
        #line = "valdataset/" + "val_dataset/" + "image1/" + line.strip() + ' ' + "val_dataset/" + "val_dataset/" + "image2/" + line.strip() + ' ' + "val_dataset/" + "val_dataset/" + "gt/" + line.strip() + '\n'
        line = "test_dataset/" + "A/" + line.strip() + ' ' + "test_dataset/" + "B" + line.strip() + ' ' + "test_dataset/" + "label/" + line.strip() + '\n'
        # 将处理后的行写入目标文件
        target_file.write(line)

print("复制完成，结果已保存到output.txt文件中")
