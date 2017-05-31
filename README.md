# CNNdel



1. Run a tool named SVfeature(https://github.com/zz-zigzag/SVfeature) to get the deletion candidates. Compile a c file(https://github.com/zz-zigzag/bioinformatics/blob/master/tools/verify-deletion.c)
2. Download Keras(https://github.com/fchollet/keras) and install.
3. system requirements
 Python 3.5
 Keras 
 NumPy >= 1.8.2
4. Installing CNNdel (https://github.com/JingWCrystal/CNNdel)
5. Running CNNdel.
 Pay attention to the path of input data in the code, you must change it to yours.
 command 'python CNNdel.py -h' for help 

 Usage: [options] train_data train_tag test_path 
 
 sample: python3 CNNdel.py CNNtrain.data CNN_train_lable.txt /home/wj/DL/test/NA19984.chrom20.DLtest1.data -o res.txt
 
 options:
 
 -l learning rate: set learning rate of the CNNs
  -- default 0.1 
  
 -b batch: set batch of train input
  -- default 32 
  
 -e epoch: set epoch of train input
  -- default 50 
  
 -a activation: set relu as activation of CNNs
  -- default tanh 
  
 -s shuffle: set shuffle mode open for train process
 -- default no shuffle 
 
 -n neurons : set number of neurons in flatten layer
  -- default 256.
  
 -o out_file: output file


