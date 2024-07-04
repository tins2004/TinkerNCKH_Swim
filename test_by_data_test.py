import numpy as np
import pandas as pd
# import tflite_runtime.interpreter as tflite
import os

data_path = './data/datatest_0to20/'
model_path = './model/Model_Swiming.tflite'
label_test = 2
col = ["ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2", "MAG_0", "MAG_1", "MAG_2", "PRESS", "label"]
col_no_label = ["ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2", "MAG_0", "MAG_1", "MAG_2", "PRESS"]

#cào dữ liệu
main_dataframe = pd.DataFrame()
user_list = os.listdir(data_path)
for i in range(10, 21):
  folder_data = data_path + str(i)
  print(folder_data)
  csv_file = os.listdir(folder_data)
  for file in csv_file:
    if file.endswith('.csv'):
      df = pd.read_csv(folder_data + '/' + file)
      main_dataframe = pd.concat([main_dataframe, df])

#xữ lý label
main_dataframe = main_dataframe.loc[(main_dataframe['label'] != 6.0) & (main_dataframe['label'] != -1.0)]
labels = set(main_dataframe['label'])
main_dataframe = main_dataframe[col]

#chọn label muốn kiểm tra
main_dataframe["label"] = main_dataframe["label"].astype(str)
if main_dataframe["label"].str.contains(str(label_test)).any():
  data = main_dataframe[main_dataframe["label"].str.contains(str(label_test))]
  main_dataframe = data
else:
  print("no label")



#hàm chia dữ liệu thành từng đoạn 180
def find_multiples_of_180(len):
  multiples = []
  for i in range(0, len+1, 180):
      multiples.append(i)
  return multiples

#Lọc dữ liệu thành (X, 180, 10)
def test_generator(dataframe):
  X_val = np.zeros((int(dataframe.shape[0] / 180), 180, dataframe[col_no_label].shape[1]))

  for i in range(int(dataframe.shape[0] / 180)):
    pos_result = find_multiples_of_180(main_dataframe.shape[0])
    pos = np.random.randint(len(pos_result) - 1)

    X_val[i, :, :] = dataframe[col_no_label][pos_result[pos]:pos_result[pos + 1]]

  return X_val

X_test = test_generator(main_dataframe)

#Chia dữ liệu thành (X, 180, 10, 1)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

#Hàm thử dữ liệu
interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()
def run_inference(data_test):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    expected_input_shape = input_details[0]['shape']

    if not np.array_equal(data_test.shape, expected_input_shape):
        print("Data shape incompatible with model input. Please check data preprocessing or model expectations.")
        return None

    data_test = data_test.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], data_test)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

#Thử toàn bộ dữ liệu
count_label_0 = 0
count_label_1 = 0
count_label_2 = 0
count_label_3 = 0
count_label_4 = 0
for i in range(len(X_test)):
  data_test = X_test[i]
  data_test = data_test.reshape((1, X_test.shape[1], X_test.shape[2], X_test.shape[3]))

  result = run_inference(data_test)
  max_value, max_index = max(enumerate(result[0]), key=lambda x: x[1])

  if max_value == 0:
    count_label_0 = count_label_0 + 1
  elif max_value == 1:
    count_label_1 = count_label_1 + 1
  elif max_value == 2:
    count_label_2 = count_label_2 + 1
  elif max_value == 3:
    count_label_3 = count_label_3 + 1
  elif max_value == 4:
    count_label_4 = count_label_4 + 1

#In kết quả
print("count_label_0")
print(count_label_0)

print("count_label_1")
print(count_label_1)

print("count_label_2")
print(count_label_2)

print("count_label_3")
print(count_label_3)

print("count_label_4")
print(count_label_4)