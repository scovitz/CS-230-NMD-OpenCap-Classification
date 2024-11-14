import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import pandas as pd

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('class_info.csv')
Class = df['Class'].values
ID = df['ID'].values # Don't put into the model

df2 = pd.read_csv('video_features.csv')
mwrt_ankle_elev = df2['10mwrt_ankle_elev'].values
mwrt_com_sway = df2['10mwrt_com_sway'].values
mwrt_mean_max_ka = df2['10mwrt_mean_max_ka'].values
mwrt_mean_ptp_hip_add = df2['10mwrt_mean_ptp_hip_add'].values
mwrt_speed = df2['10mwrt_speed'].values
mwrt_stride_len = df2['10mwrt_stride_len'].values
mwrt_stride_time = df2['10mwrt_stride_time'].values
mwrt_trunk_lean = df2['10mwrt_trunk_lean'].values
mwt_ankle_elev = df2['10mwt_ankle_elev'].values
mwt_com_sway = df2['10mwt_com_sway'].values
mwt_mean_max_ka = df2['10mwt_mean_max_ka'].values
mwt_mean_ptp_hip_add = df2['10mwt_mean_ptp_hip_add'].values
mwt_speed = df2['10mwt_speed'].values
mwt_stride_len = df2['10mwt_stride_len'].values
mwt_stride_time = df2['10mwt_stride_time'].values
mwt_trunk_lean = df2['10mwt_trunk_lean'].values
xsts_lean_max = df2['5xsts_lean_max'].values
xsts_stance_width = df2['5xsts_stance_width'].values
xsts_time_5 = df2['5xsts_time_5'].values
arm_rom_rw_area = df2['arm_rom_rw_area'].values
brooke_max_ea_at_max_min_sa = df2['brooke_max_ea_at_max_min_sa'].values
brooke_max_mean_sa = df2['brooke_max_mean_sa'].values
brooke_max_min_sa = df2['brooke_max_min_sa'].values
brooke_max_sa_ea_ratio = df2['brooke_max_sa_ea_ratio'].values
curls_max_mean_ea = df2['curls_max_mean_ea'].values
curls_min_max_ea = df2['curls_min_max_ea'].values
jump_max_com_vel = df2['jump_max_com_vel'].values
toe_stand_int_com_elev = df2['toe_stand_int_com_elev'].values
toe_stand_int_mean_heel_elev = df2['toe_stand_int_mean_heel_elev'].values
toe_stand_int_trunk_lean = df2['toe_stand_int_trunk_lean'].values
toe_stand_mean_int_aa = df2['toe_stand_mean_int_aa'].values
tug_cone_time = df2['tug_cone_time'].values
tug_cone_turn_avel = df2['tug_cone_turn_avel'].values
tug_cone_turn_max_avel = df2['tug_cone_turn_max_avel'].values

x = df2[['10mwrt_ankle_elev', '10mwrt_com_sway', '10mwrt_mean_max_ka', '10mwrt_mean_ptp_hip_add', '10mwrt_speed', '10mwrt_stride_len', '10mwrt_stride_time','10mwrt_trunk_lean','10mwt_ankle_elev',...
'10mwt_com_sway','10mwt_mean_max_ka','10mwt_mean_ptp_hip_add', '10mwt_speed','10mwt_stride_len','10mwt_stride_time','10mwt_trunk_lean','5xsts_lean_max',...
'5xsts_stance_width','5xsts_time_5','arm_rom_rw_area','brooke_max_ea_at_max_min_sa','brooke_max_mean_sa','brooke_max_min_sa','brooke_max_sa_ea_ratio',...
'curls_max_mean_ea','curls_min_max_ea','jump_max_com_vel','toe_stand_int_com_elev','toe_stand_int_mean_heel_elev','toe_stand_int_trunk_lean','toe_stand_mean_int_aa',...
'tug_cone_time','tug_cone_turn_avel','tug_cone_turn_max_avel']].values
y = df['Class'].values

# First Split: 80% train, 20% remaining
train_size_exact = 129
X_train, X_rem, y_train, y_rem = train_test_split(x, y, train_size=train_size_exact, shuffle=True)

# Debug: Print sizes after the first split
print("Before split:")
print("x:", x.shape)
print("y:", y.shape)
print("After first split:")
print("x_train size:", x_train.shape)  # Should be ~129
print("x_rem size:", x_rem.shape)      # Should be ~40
print("y_train size:", y_train.shape)  # Should be ~129
print("y_rem size:", y_rem.shape)      # Should be ~40

# Second Split: Remaining 20% split into 50% validation, 50% test
X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, train_size=0.5, shuffle=True)

# Debug: Print sizes after the second split
print("After second split:")
print("x_val size:", x_val.shape)      # Should be ~20
print("x_test size:", x_test.shape)    # Should be ~20
print("y_val size:", y_val.shape)      # Should be ~20
print("y_test size:", y_test.shape)    # Should be ~20


# Model definition
model = Sequential()

# First GRU layer
model.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # Dropout for regularization

# Second GRU layer (optional)
model.add(GRU(units=64, return_sequences=False))

# Fully connected Dense layer
model.add(Dense(units=32, activation='relu'))

# Output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))

# Evaluate the model
model.evaluate(X_test, y_test)

# Predictions
predictions = model.predict(X_new)
