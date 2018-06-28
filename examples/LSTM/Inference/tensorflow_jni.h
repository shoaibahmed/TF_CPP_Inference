// The methods are exposed to Java to allow for interaction with the native
// Tensorflow code. See
// tensorflow/examples/android/src/org/tensorflow/TensorflowClassifier.java
// for the Java counterparts.

#ifndef ORG_TENSORFLOW_JNI_TENSORFLOW_JNI_H_  // NOLINT
#define ORG_TENSORFLOW_JNI_TENSORFLOW_JNI_H_  // NOLINT

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define TENSORFLOW_METHOD(METHOD_NAME) \
  Java_org_epiwear_app_TensorflowClassifier_##METHOD_NAME  // NOLINT

JNIEXPORT jint JNICALL
TENSORFLOW_METHOD(initializeTensorflow)(
    JNIEnv* env, jobject thiz, jobject java_asset_manager,
    jstring modelECG, jstring modelBreathingRate, jstring modelMotionClassifier,
    jstring externalStorageLoc);

JNIEXPORT jfloat JNICALL
TENSORFLOW_METHOD(computeAnomalyScore)(
    JNIEnv* env, jobject thiz, jfloat accRMS, jfloat eda, jfloat ecg, jfloat br);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ORG_TENSORFLOW_JNI_TENSORFLOW_JNI_H_  // NOLINT
