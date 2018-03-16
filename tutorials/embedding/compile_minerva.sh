INTERPRETER="python3.5"
TF_INC=$($INTERPRETER -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_LIB=$($INTERPRETER -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
NSYNC_INC=$TF_INC"/external/nsync/public"
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC  -I $NSYNC_INC -O2 # -L$TF_LIB -ltensorflow_framework
