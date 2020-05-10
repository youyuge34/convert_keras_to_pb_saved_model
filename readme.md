# Convert Keras model into constant single pb file

> Only for tensorflow 1.x!

It is just a tutorial for converting a running Keras model into a single pb file with tf.saved_model API, and without any intermediate convertions.

If u just want to convert .h5 Keras model file into pb format, refer to: https://github.com/amir-abdi/keras_to_tensorflow

# Requirements
```
python >= 3.5
tensorflow >= 1.4.0, <= 1.15.x
Keras >= 2.1.3
absl
numpy
```

# Usage
- Change the hyperparameters in the `main()` function as u wish:
```
    LOG = True
    logging.set_verbosity(logging.INFO)
    CHECK_VALUE = True  # to check the predict values are the same or not, between origin Keras model and output pb model
    OUT_PUTDIR = 'output_single_pb'
    
    x_input = prepare_test_img_input(img_path='images/34rews.jpg')

    save(use_saved_model=True)  # use the tf.saved_model API instead of just writting constant graph_def into file
    # test(use_saved_model=True)  # To test the generated pb file
```

- Change the signatures as u wish to adapt your tensorflow serving service.

- run the command `python saved_keras_model_2_single_pb.py`.

- Then if u want to test the generated pb file, just uncomment the `test()` function in above codes and comment `save()` function.

# FAQ
**Why u need to check the predict values are the same or not, between origin Keras model and output pb model?**

**A:** There is a bug if call K.set_learning_phase(0) when using keras.layer.BN and `convert_variables_to_constants()` together. Refer to :https://github.com/amir-abdi/keras_to_tensorflow/issues/109


**Why u don't call `K.set_learning_phase(0)` ?**

**A:** Explained as above.