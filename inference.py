import tensorflow as tf
import librosa
import logging
import numpy as np
from typing import Tuple, List
from scipy.io.wavfile import write
import os
import time
import re
import onnxruntime as ort
from typing import Dict, List, Any, Optional, Union
from denoiser.dsp import convert_audio


# Create Inference Class
class InferencePipeline:
    """
        Class with the complete inference pipeline and is able to handle keras, tflite and onnx model.
    """
    # INIT
    def __init__(self, model_path: str, output_dir: str):

        self.model_path = model_path
        self.model_type = self._detect_model_type()
        self.model = self._load_model()

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.target_sr = 16000
        self.batch_size = 12000

        self.filename = None
        self.data = None
        self.data_sr = None
        self.data_duration = None
        self.data_path = None
        self.data_batches, self.diff = None, None
        self.output_path = None
        self.output = None
        self.predictions = None
        self.data_rs = None
        self.data_rs_onnx = None
       
    # RUN
    def run(self, data_path:str):
        logging.info(f"Running Inference on {os.path.basename(data_path)}........")
        if data_path.endswith(".wav"):
            time_dict = {}
            # Get Data Path
            self.data_path = data_path
            # Get Filename
            # Time to load the data
            t_load_start = time.time()
            self.filename = os.path.basename(self.data_path)
            # Load Data
            self._load_data()
            t_load = time.time() - t_load_start
            t_load = round(t_load, 3)
            
            # time taken to preprocess data
            t_preprocess_start = time.time()
            # Preprocess Data
            self._preprocess()
            t_preprocess = time.time() - t_preprocess_start
            t_preprocess = round(t_preprocess, 3)
            
            # time taken to predict
            t_predict_start = time.time()
            # Predict Data
            self._predict()
            t_predict = time.time() - t_predict_start
            t_predict = round(t_predict, 3)
            # time taken to postprocess
            t_postprocess_start = time.time()
            # Postprocess Data
            self._postprocess()
            t_postprocess = time.time() - t_postprocess_start
            t_postprocess = round(t_postprocess, 3)
            # time to save output
            t_save_start = time.time()
            # Save Output
            self._save_output()
            t_save_output = time.time() - t_save_start
            t_save_output = round(t_save_output, 3)
            t_total = t_load + t_preprocess + t_predict + t_postprocess + t_save_output
            time_dict["t_load"] = t_load
            time_dict["t_preprocess"] = t_preprocess
            time_dict["t_predict"] = t_predict
            time_dict["t_postprocess"] = t_postprocess
            time_dict["t_save_output"] = t_save_output
            time_dict["t_total"] = t_total

            return self.data_duration, self.output_path, time_dict
        else:
            raise ValueError(f"Only .wav file supported for inference.")
        
    # Detect Model  - TFLITE, KERAS OR ONNX
    def _detect_model_type(self):
        logging.info(f"Detecting model type.....")
        if self.model_path.endswith("keras"):
            logging.info(f"Model Type sucessfully detected as KERAS")
            return "keras"
        elif self.model_path.endswith(".tflite"):
            logging.info(f"Model Type sucessfully detected as TFLITE")
            return "tflite"
        elif self.model_path.endswith(".onnx"):
            logging.info(f"Model Type sucessfully detected as ONNX")
            return "onnx"
        elif self.model_path.endswith(".pth"):
            logging.info(f"Model Type sucessfully detected as pth (PyTorch)")
            return "pytorch"
        else:
            raise ValueError("Unsopported model format. Use .keras or .tflite.")
        
    # Load Model - TFLITE, KERAS, or ONNX, or Pytorch
    def _load_model(self):
        logging.info(f"Loading {self.model_type} from {self.model_path}...")
        if self.model_type == "keras":
            model = tf.keras.models.load_model(self.model_path)
        elif self.model_type == "tflite":
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            model = interpreter
        elif self.model_type == "onnx":
            # Load the ONNX Runtime session for the quantized model
            ort_session = ort.InferenceSession(self.model_path)
            model = ort_session
        elif self.model_type == "pytorch":
            model = torch.load(self.model_path, weights_only=False)
        logging.info(f"Loaded the {self.model_type} successfully")
        return model
    
    # Load Data
    def _load_data(self):
        logging.info(f"Loading data from {self.data_path}")
        # Load Data
        self.data, self.data_sr = librosa.load(self.data_path, sr=48000)
        self.data_duration = librosa.get_duration(y=self.data, sr=self.data_sr)
        print(f"Data shape: {self.data.shape}, Data Sample Rate: {self.data_sr} Hz, Data Duration: {self.data_duration}")

    # Preprocess data
    def _preprocess(self):
        logging.info(f"Preprocessing the data....")
    
        # Resample Audio
        if self.data_sr != self.target_sr:
            # wav = librosa.resample(y=wav, orig_sr=sr, target_sr=target_sr)
            self.data_rs = librosa.resample(self.data, orig_sr=self.data_sr, target_sr=self.target_sr)
        else:
            self.data_rs = self.data
        logging.info(f"Data Resampled Shape: {self.data_rs.shape}")

        # if model type is keras or tflite convert to tensor
        logging.info("Data Conversion based on model type....")
        if self.model_type.lower() in ["keras", "tflite"]:
            # Convert Numpy to Tensor
            data_tensor = tf.convert_to_tensor(self.data_rs)
            # Create batches
            self.data_batches, self.diff = self.__create_batches(data_tensor)
        elif self.model_type.lower() in ["onnx"]:
            if self.data_rs.ndim == 1:
                self.data_rs = np.expand_dims(self.data_rs, axis=0)
            # Prepare the input for the ONNX model
            self.data_rs_onnx = np.expand_dims(self.data_rs, axis=0).astype(np.float32)
        else: 
            # For Pytorch model we need a tensor
            self.data_rs_pth = torch.tensor(self.data_rs)
            print(f"Torch Tensor shape for PyTorch model: {self.data_rs_pth.shape}")
            # Denoiser Convert Audio -> wav, from_samplerate, to_samplerate, channels
            self.data_rs_pth = convert_audio(wav=self.data_rs_pth.cuda(), from_samplerate=self.target_sr, to_samplerate=self.target_sr, channels=1)
            print(f"Torch Tensor shape for PyTorch model (after convert audio function call): {self.data_rs_pth.shape}")

        logging.info(f"Preprocessing Successful.")

    # Predict
    def _predict(self):
        logging.info(f"Running prediction using {self.model_type} model...")
        # If keras model
        if self.model_type == "keras":
            self.predictions = self.model.predict(self.data_batches)

        elif self.model_type == "tflite":
            interpreter = self.model
            input_details = interpreter.get_input_details()
            input_expected_shape = input_details[0]['shape']
            input_index = input_details[0]['index']
            output_index = interpreter.get_output_details()[0]['index']

            
            print(f"Expected input shape: {input_expected_shape}")
            print(f"Actual input shape before setting tensor: {self.data_batches.shape}")

            preds = []
            for i in self.data_batches:
                print(f"Actual input shape of single tensor: {i.shape}")
                i = tf.reshape(i, input_expected_shape)
                print(f"Converted input shape of single tensor: {i.shape}")
                interpreter.set_tensor(input_index, i)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_index)
                preds.append(predictions)
            
            self.predictions = tf.squeeze(tf.stack(preds, axis=1))
            self.predictions = tf.expand_dims(self.predictions, axis=-1)
        elif self.model_type == "onnx":
            # Add ONNX inference
             # Perform inference with the ONNX model
            self.predictions = self.model.run(None, {"input": self.data_rs_onnx})
        else:
            # For Pytorch Model
            with torch.no_grad():
                self.predictions = model(self.data_rs_pth[None])[0]

    # Post Process Data
    def _postprocess(self):
        logging.info("Postprocessing the predictions....")
        # TODO: postprocessing code
        # Postprocessing based on TFLITE, KERAS or ONNX Model
        if self.model_type.lower() in ["keras", "tflite"]:
            # Reshape
            print(f"Predictions Type and Shape: {type(self.predictions)}, {self.predictions.shape}")
            self.output = tf.reshape(self.predictions[:-1], ((self.predictions.shape[0] - 1)*self.predictions.shape[1], 1))
            print(f"Output Type and Shape: {type(self.output)}, {self.output.shape}")
            # Concat the Last Batch
            print(f"Uneven Batch Shape: {self.diff}, {self.predictions[-1][-self.diff:].shape}")
            self.output = tf.concat((self.output, self.predictions[-1][-self.diff:]), axis=0)
            # Convert Tensor to numpy
            self.output = self.output.numpy()
            # Reshape the numpy
            self.output = np.squeeze(self.output, -1)
            logging.info("Postprocessing successful.")
        elif self.model_type.lower() in ["onnx"]:
            # Add data postprocessing for onnx model predictions
            self.output = self.predictions[0]
            # Flatten to 1D array if needed
            self.output =  self.output.squeeze()
            logging.info("Postprocessing sucessful.")
        else:
            # For Pytorch model which returns a torch tensor
            self.output = self.predictions
            self.output = self.output.cpu().numpy()
            self.output = self.output.squeeze()
            logging.info("Postprocessing successful.")

    # Save Output
    def _save_output(self):
        # Set Output Path
        model_name = os.path.basename(self.model_path)
        match = re.search(r"d(\d{8})_t\d{6}_e(\d+)_b(\d+)", model_name)
        if match: 
            train_date, epochs, _ = match.groups()
            model_abbr = f"d{train_date}_e{epochs}"
        else:
            model_abbr = model_name[:17] # Fallback
            model_abbr = model_abbr.rstrip("_")
        output_dir_final = os.path.join(self.output_dir, model_abbr)
        os.makedirs(output_dir_final, exist_ok=True)
        self.output_path = os.path.join(output_dir_final, self.model_type+"_output_"+self.filename)
        logging.info(f"Saving the output to the {output_dir_final}")
        # Save output to a file
        sample_rate = self.target_sr  # Hz
        audio_data = self.output
        print(f"Output Audio Data Shape Before Saving: {audio_data.shape}, Audio Data Sample Rate Before Saving: {sample_rate}")
        
        # Save as WAV
        write(self.output_path, sample_rate, audio_data)

        logging.info(f"Successfully saved the output to the {self.output_path}")

    # Batch Creation - Specific to Keras and TFLITE Model
    def __create_batches(self, data_tensor: tf.Tensor) -> Tuple[tf.Tensor, int]:
        logging.info("Creating Batches for Inference...")
        data_len = data_tensor.shape[0]
        batches = []
        for i in range(0, data_len - self.batch_size, self.batch_size):
            batches.append(self.data[i:i+self.batch_size])
        
        batches.append(self.data[-self.batch_size:])
        diff = data_len - (i + self.batch_size)

        logging.info("Batches created sucessfully.")
        return tf.stack(batches), diff

class BatchInferencePipeline(InferencePipeline):
    
    def __init__(self, model_path: str, output_dir: str):
        super().__init__(model_path=model_path, output_dir=output_dir)
    
    def run_batch(self, data_paths: List[str]) -> Optional[Dict[str, Union[str, Dict[str, Any]]]]:
        results = {}
        for data_path in data_paths:
            try:
                data_duration, output_path, time_dict = self.run(data_path=data_path)
                results[data_path] = {"output_path": output_path, "data_duration": data_duration,"inference_time_dict": time_dict}
                logging.info(f"Successfully processed: {data_path} -> {output_path}")
            except Exception as e:
                logging.error(f"Failed to process {data_path}: {str(e)}")
                results[data_path] = None

        return results



        
    
