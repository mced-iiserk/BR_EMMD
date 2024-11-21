import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split


def AutoEncoder(inShape,layers,actfs):
	# input layer
	INPUT = keras.layers.Input(shape=(inShape,))
	inL = INPUT
	
	# encoder
	for i in range(len(actfs)-1):
		l = layers[i]
		actf = actfs[i]
		inL = keras.layers.Dense(l, activation=actf)(inL)

	# latent space
	latent = keras.layers.Dense(layers[-1], activation=actfs[-1])(inL)
	inL = keras.layers.Dense(layers[-2], activation=actfs[-2])(latent)

	# decoder
	for i in range(len(actfs)-2):
		l = layers[::-1][2+i]
		actf = actfs[::-1][2+i]
		inL = keras.layers.Dense(l, activation=actf)(inL)

	# output layer
	OUTPUT = keras.layers.Dense(inShape, activation='relu')(inL)

	# autoencoder
	ae = keras.models.Model(INPUT, OUTPUT, name='autoencoder')
	return ae


csv_file = input("enter the file: ")
data = np.loadtxt(csv_file,skiprows=1,delimiter=',')
master_data = data[:,275:-1]
ids = data[:,-1]



min_ = []
max_ = []

x= []
for i in range(378):
    MIN = np.min(master_data[:,i])
    MAX = np.max(master_data[:,i])
    min_.append(MIN)
    max_.append(MAX)
    scale = (master_data[:,i]-MIN)/(MAX-MIN)
    x.append(scale)
x = np.array(x)
x = x.T

np.save("MIM.npy",min_)
np.save("MAX.npy",max_)

index = np.arange(0,350000)
np.random.shuffle(index)
np.save("shuffled_index.npy",index)

training_data = x[:350000,:]
training_ids  = ids[:350000]

train_idx = index[:332500] # test split = 5%
test_idx  = index[332500:]

X_train = training_data[train_idx,:]
X_test  = training_data[test_idx,:]
y_train = ids[train_idx]
y_test  = ids[test_idx]


#X_train, X_test, y_train, y_test = train_test_split(x, ids, test_size=0.05, random_state=2024)

#AE = AutoEncoder(inShape=totftr,layers=[1024,512,128,36,12,4],actfs=['tanh','relu','tanh','relu','tanh','relu'])
AE = AutoEncoder(inShape=378,layers=[128,32,8,2],actfs=['relu','tanh','tanh','tanh'])
print(AE.summary())
AE.compile(optimizer=keras.optimizers.Adam(), loss = keras.losses.MeanSquaredError())

#exit()
results = [[],[]]
#InOut = np.load(inoutB)
#normd = (InOut-MIN)/(MAX-MIN)
#result = AE.fit(normd, normd, validation_split=0.1, batch_size=25, epochs=500)		# converged loss 5e-6
result = AE.fit(X_train, X_train, validation_split=0.05, batch_size=256, epochs=1000)
trn_loss = result.history['loss']
val_loss = result.history['val_loss']
results[0].append(trn_loss)
results[1].append(val_loss)

r2w = np.array(results)
np.save("AE_27x27_dist_results.npy",r2w)
AE.save('AE_27x27_dist.h5')
