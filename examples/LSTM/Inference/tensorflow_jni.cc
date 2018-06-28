#include "tensorflow/examples/custom/epiwear/jni/tensorflow_jni.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include <jni.h>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>
#include <cmath>
#include <fstream>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/examples/custom/epiwear/jni/jni_utils.h"

#define DEBUG 1

// Global variables that holds the Tensorflow classifier.
static std::unique_ptr<tensorflow::Session> sessionECG;
static std::unique_ptr<tensorflow::Session> sessionBreathingRate;
static std::unique_ptr<tensorflow::Session> sessionMotionClassifier;

static float trainDataECGMean = 1.68502;
static float trainDataECGStd = 0.0807502;

static float trainDataBreathingRateMean = 1.33006;
static float trainDataBreathingRateStd = 0.231518;

static float trainDataEDAMean = 1.84838e-06;
static float trainDataEDAStd = 8.30807e-07;

static float trainDataMotionMin = 37750.7273572;
static float trainDataMotionMax = 56251.6801794;

static int standardDeviationMultiplierEDA = 3;
float upperBoundEDA = trainDataEDAMean + (standardDeviationMultiplierEDA * trainDataEDAStd);
float lowerBoundEDA = trainDataEDAMean - (standardDeviationMultiplierEDA * trainDataEDAStd);

static int inputSize = 3;
static int inputSizeMotion = 33;
static int inputTimestamps = 10;
static int inputTimestampsMotion = 33;
static int numberOfHiddenUnits = 16;

int inputLength = inputSize * inputTimestamps;
int inputLengthMotion = inputSizeMotion * inputTimestampsMotion;

float accAnomalyScore;
float edaAnomalyScore;
float ecgAnomalyScore;
float brAnomalyScore;
float seizureScore;

static int num_runs = 0;

static std::ofstream dataOutputFile;
static std::string externalStoragePath;
static std::string dataOutputFileName = "/EpiWearNativeOutput.txt";

tensorflow::Tensor input_tensor_ecg(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({1, inputTimestamps, inputSize}));
auto input_tensor_mapped_ecg = input_tensor_ecg.tensor<float, 3>();

tensorflow::Tensor input_tensor_breathingRate(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({1, inputTimestamps, inputSize}));
auto input_tensor_mapped_breathingRate = input_tensor_breathingRate.tensor<float, 3>();

tensorflow::Tensor input_tensor_motion(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({1, inputTimestamps, inputSize}));
auto input_tensor_mapped_motion = input_tensor_motion.tensor<float, 3>();

tensorflow::Tensor input_state_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({1, 2 * numberOfHiddenUnits}));

std::vector<tensorflow::Tensor> output_tensors;
std::vector<std::string> output_names_ecg({"Anomaly_ECG_Ops/output_node:0"});
std::vector<std::string> output_names_breathingRate({"Anomaly_BreathingRate_Ops/output_node:0"});
std::vector<std::string> output_names_motion({"Motion_Classifier_Ops/output_node:0"});

tensorflow::Status s;

static bool g_compute_graph_initialized = false;

using namespace tensorflow;

inline static int64 CurrentThreadTimeUs() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

JNIEXPORT jint JNICALL
TENSORFLOW_METHOD(initializeTensorflow)(
    JNIEnv* env, jobject thiz, jobject java_asset_manager,
    jstring modelECG, jstring modelBreathingRate, jstring modelMotionClassifier,
    jstring externalStorageLoc) {
  //MutexLock input_lock(&g_compute_graph_mutex);
  if (g_compute_graph_initialized) {
    LOG(INFO) << "Compute graph already loaded. skipping.";
    return 0;
  }

  const char* const modelECG_cstr = env->GetStringUTFChars(modelECG, NULL);
  const char* const modelBreathingRate_cstr = env->GetStringUTFChars(modelBreathingRate, NULL);
  const char* const modelMotionClassifier_cstr = env->GetStringUTFChars(modelMotionClassifier, NULL);

  const char* externalStorageLocation = env->GetStringUTFChars(externalStorageLoc, NULL);
  externalStoragePath.assign(externalStorageLocation, strlen(externalStorageLocation));

  LOG(INFO) << "Loading Tensorflow.";

  LOG(INFO) << "Making new SessionOptions.";
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  LOG(INFO) << "Got config, " << config.device_count_size() << " devices";

  sessionECG.reset(tensorflow::NewSession(options));
  sessionBreathingRate.reset(tensorflow::NewSession(options));
  sessionMotionClassifier.reset(tensorflow::NewSession(options));
  LOG(INFO) << "Sessions created";

  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created";

  AAssetManager* const asset_manager =
      AAssetManager_fromJava(env, java_asset_manager);
  LOG(INFO) << "Acquired AssetManager";

  // Reading ECG Model file
  LOG(INFO) << "Reading ECG file to proto: " << modelECG_cstr;
  ReadFileToProto(asset_manager, modelECG_cstr, &tensorflow_graph);

  LOG(INFO) << "Creating session for ECG";
  s = sessionECG->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
    return -1;
  }

  // Clear the proto to save memory space.
  tensorflow_graph.Clear();
  LOG(INFO) << "Tensorflow graph for ECG loaded from: " << modelECG_cstr;

  // Reading Breathing Rate Model file
  LOG(INFO) << "Reading Breathing Rate file to proto: " << modelBreathingRate_cstr;
  ReadFileToProto(asset_manager, modelBreathingRate_cstr, &tensorflow_graph);

  LOG(INFO) << "Creating session for Breathing Rate";
  s = sessionBreathingRate->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
    return -1;
  }

  // Clear the proto to save memory space.
  tensorflow_graph.Clear();
  LOG(INFO) << "Tensorflow graph for Breathing Rate loaded from: " << modelBreathingRate_cstr;

  // Reading Motion Classifier Model file
  LOG(INFO) << "Reading Motion Classification file to proto: " << modelMotionClassifier_cstr;
  ReadFileToProto(asset_manager, modelMotionClassifier_cstr, &tensorflow_graph);

  LOG(INFO) << "Creating session for Motion Classification";
  s = sessionMotionClassifier->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
    return -1;
  }

  // Clear the proto to save memory space.
  tensorflow_graph.Clear();
  LOG(INFO) << "Tensorflow graph for Motion Classification loaded from: " << modelMotionClassifier_cstr;

  // Create input state tensor
  auto input_state_tensor_mapped = input_state_tensor.tensor<float, 2>();

  LOG(INFO) << "Tensorflow: Copying Data to Input_State Tensor";
  for (int i = 0; i < 2 * numberOfHiddenUnits; ++i) {
    input_state_tensor_mapped(0, i) = (float) 0.0;
  }

  dataOutputFileName = externalStoragePath + dataOutputFileName;
  g_compute_graph_initialized = true;

  return 0;
}


static float computeAnomalyScoreMotion(float accRMS)
{
  // Normalize data
  accRMS = (accRMS - trainDataMotionMin) / (float) (trainDataMotionMax - trainDataMotionMin);

  // Get system prediction
  std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors(
    {{"Motion_Classifier/Input_X:0", input_tensor_motion},
     {"Motion_Classifier/Input_State:0", input_state_tensor}});

  s = sessionMotionClassifier->Run(input_tensors, output_names_motion, {}, &output_tensors);

  // Shift the whole tensor to accomodate new sample
  int rowIndex = 0;
  int colIndex = 0;
  for (int i = 0; i < inputLength; ++i) {
    if(colIndex != 0)
      input_tensor_mapped_motion(0, rowIndex, colIndex - 1) = input_tensor_mapped_motion(0, rowIndex, colIndex);
    else
    {
      if(rowIndex != 0)
        input_tensor_mapped_motion(0, rowIndex - 1, inputSize - 1) = input_tensor_mapped_motion(0, rowIndex, colIndex);
    }
    
    colIndex++;

    if(colIndex >= inputSize)
    {
      colIndex = 0;
      rowIndex++;
    }
  }

  // Insert the latest sample
  input_tensor_mapped_motion(0, inputTimestamps - 1, inputSize - 1) = accRMS;

  // Return if the tensor is not yet full
  if(num_runs < inputLength)
  {
    return 0;
  }

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
    return -1;
  }

  tensorflow::Tensor* output = &output_tensors[0];

  Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                          Eigen::Aligned> prediction = output->flat<float>();

  assert(prediction.size() == 2);

  if(DEBUG)
    VLOG(0) << " Seizure Motion Probability: " << prediction(1);

  return prediction(1);
}


static float computeAnomalyScoreEDA(float eda)
{
  float mse = 0;

  if(eda > upperBoundEDA)
    mse = eda - upperBoundEDA;
  else if (eda < lowerBoundEDA)
    mse = lowerBoundEDA - eda;

  if(DEBUG)
    VLOG(0) << " EDA MSE: " << mse;

  return mse;
}


static float computeAnomalyScoreECG(float ecg)
{
  float mse = 0;

  // Normalize data
  ecg = (ecg - trainDataECGMean) / (float) trainDataECGStd;

  // Get system prediction
  std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors(
    {{"Anomaly_ECG/Input_X:0", input_tensor_ecg},
     {"Anomaly_ECG/Input_State:0", input_state_tensor}});

  s = sessionECG->Run(input_tensors, output_names_ecg, {}, &output_tensors);

  // Shift the whole tensor to accomodate new sample
  int rowIndex = 0;
  int colIndex = 0;
  for (int i = 0; i < inputLength; ++i) {
    if(colIndex != 0)
      input_tensor_mapped_ecg(0, rowIndex, colIndex - 1) = input_tensor_mapped_ecg(0, rowIndex, colIndex);
    else
    {
      if(rowIndex != 0)
        input_tensor_mapped_ecg(0, rowIndex - 1, inputSize - 1) = input_tensor_mapped_ecg(0, rowIndex, colIndex);
    }
    
    colIndex++;

    if(colIndex >= inputSize)
    {
      colIndex = 0;
      rowIndex++;
    }
  }

  // Insert the latest sample
  input_tensor_mapped_ecg(0, inputTimestamps - 1, inputSize - 1) = ecg;

  // Return if the tensor is not yet full
  if(num_runs <= inputLength)
    return mse;

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
    return -1;
  }

  tensorflow::Tensor* output = &output_tensors[0];

  Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                          Eigen::Aligned> prediction = output->flat<float>();

  assert(prediction.size() == 1);

  // Return MSE
  mse = pow(prediction(0) - ecg, 2);

  if(DEBUG)
    VLOG(0) << " ECG MSE: " << mse;

  return mse;
}


static float computeAnomalyScoreBR(float br)
{
  float mse = 0;

  // Normalize data
  br = (br - trainDataBreathingRateMean) / (float) trainDataBreathingRateStd;

  // Get system prediction
  std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors(
    {{"Anomaly_BreathingRate/Input_X:0", input_tensor_breathingRate},
     {"Anomaly_BreathingRate/Input_State:0", input_state_tensor}});

  s = sessionBreathingRate->Run(input_tensors, output_names_breathingRate, {}, &output_tensors);

  // Shift the whole tensor to accomodate new sample
  int rowIndex = 0;
  int colIndex = 0;
  for (int i = 0; i < inputLength; ++i) {
    if(colIndex != 0)
      input_tensor_mapped_breathingRate(0, rowIndex, colIndex - 1) = input_tensor_mapped_breathingRate(0, rowIndex, colIndex);
    else
    {
      if(rowIndex != 0)
        input_tensor_mapped_breathingRate(0, rowIndex - 1, inputSize - 1) = input_tensor_mapped_breathingRate(0, rowIndex, colIndex);
    }
    
    colIndex++;

    if(colIndex >= inputSize)
    {
      colIndex = 0;
      rowIndex++;
    }
  }

  // Insert the latest sample
  input_tensor_mapped_breathingRate(0, inputTimestamps - 1, inputSize - 1) = br;

  // Return if the tensor is not yet full
  if(num_runs <= inputLength)
    return mse;

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
    return -1;
  }

  tensorflow::Tensor* output = &output_tensors[0];

  Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                          Eigen::Aligned> prediction = output->flat<float>();

  assert(prediction.size() == 1);
  
  // Return MSE
  mse = pow(prediction(0) - br, 2);

  if(DEBUG)
    VLOG(0) << " BR MSE: " << mse;

  return mse;
}


static float computeScore(float accRMS, float eda, float ecg, float br) {
  // Very basic benchmarking functionality.
  static int64 timing_total_us = 0;
  const int64 start_time = CurrentThreadTimeUs();

  // Compute individual anomaly scores
  accAnomalyScore = computeAnomalyScoreMotion(accRMS);
  // edaAnomalyScore = computeAnomalyScoreEDA(eda);
  // ecgAnomalyScore = computeAnomalyScoreECG(ecg);
  // brAnomalyScore = computeAnomalyScoreBR(br);
  
  const int64 end_time = CurrentThreadTimeUs();
  num_runs++;

  const int64 elapsed_time_inf = end_time - start_time;
  timing_total_us += elapsed_time_inf;

  if(DEBUG)
  {
    VLOG(0) << "Ran in " << elapsed_time_inf / 1000 << "ms ("
          << (timing_total_us / num_runs / 1000) << "ms avg over " << num_runs
          << " runs)";
  }

  // Open file for data writing
  dataOutputFile.open(dataOutputFileName, std::ofstream::out | std::ofstream::app);

  // seizureScore = 0.5e6 *  edaAnomalyScore + 0.25 * ecgAnomalyScore + 5 * brAnomalyScore + accAnomalyScore;
  // dataOutputFile << accAnomalyScore << " " << edaAnomalyScore << " " << ecgAnomalyScore << " " 
  //               << brAnomalyScore << " " << seizureScore << std::endl;

  seizureScore = accAnomalyScore;
  dataOutputFile << seizureScore << std::endl;

  // Close file
  dataOutputFile.close();

  return seizureScore;
}

JNIEXPORT jfloat JNICALL
TENSORFLOW_METHOD(computeAnomalyScore)(
    JNIEnv* env, jobject thiz, jfloat accRMS, jfloat eda, jfloat ecg, jfloat br) {
  // Copy image into currFrame.
  // jboolean iCopied = JNI_FALSE;
  // jfloat* mySignal = env->GetFloatArrayElements(signal, &iCopied);

  jfloat result = computeScore(accRMS, eda, ecg, br);

  // env->ReleaseFloatArrayElements(signal, mySignal, JNI_ABORT);

  return result;
}
