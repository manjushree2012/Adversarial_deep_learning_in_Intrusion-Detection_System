import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import time

from tensorflow import set_random_seed

from sklearn.preprocessing import MinMaxScaler

from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation

from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod, CarliniWagnerL2, DeepFool
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.evaluation import batch_eval

import matplotlib.pyplot as plt

# Set TF random seed to improve reproducibility
tf.set_random_seed(42)
np.random.seed(0)

if not hasattr(backend, "tf"):
        raise RuntimeError("Requires keras to be configured to use the TensorFlow backend.")










###########################################################################
                            # DATA PREPROCESSING
###########################################################################

print()
print()
print("------ DATA PREPROCESSING STAGE --------")

df = pd.read_csv('KDDTrain+.csv',  header = None) #training dataset
dft = pd.read_csv('KDDTest+.csv',  header = None) #test dataset

print("Initial training and test data shapes : ", df.shape, dft.shape)

#concat both datasets in same variable
full = pd.concat([df, dft])
assert full.shape[0] == df.shape[0] + dft.shape[0]
assert full.shape[1] == df.shape[1] == dft.shape[1]

full.columns = ['duration',
                'protocol_type',
                'service',
                'flag',
                'src_bytes',
                'dst_bytes',
                'land',
                'wrong_fragment',
                'urgent',
                'hot',
                'num_failed_logins',
                'logged_in',
                'num_compromised',
                'root_shell',
                'su_attempted',
                'num_root',
                'num_file_creations',
                'num_shells',
                'num_access_files',
                'num_outbound_cmds',
                'is_host_login',
                'is_guest_login',
                'count',
                'srv_count',
                'serror_rate',
                'srv_serror_rate',
                'rerror_rate',
                'srv_rerror_rate',
                'same_srv_rate',
                'diff_srv_rate',
                'srv_diff_host_rate',
                'dst_host_count',
                'dst_host_srv_count',
                'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate',
                'dst_host_srv_serror_rate',
                'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate',
                'label']

full.loc[full['label'] == 'buffer_overflow','label'] = 'u2r'
full.loc[full['label'] == 'loadmodule','label'] = 'u2r'
full.loc[full['label'] == 'rootkit','label'] = 'u2r'
full.loc[full['label'] == 'perl','label'] = 'u2r'
full.loc[full['label'] == 'sqlattack','label'] = 'u2r'
full.loc[full['label'] == 'xterm','label'] = 'u2r'
full.loc[full['label'] == 'ps','label'] = 'u2r'

full.loc[full['label'] == 'neptune','label'] = 'dos'
full.loc[full['label'] == 'back','label'] = 'dos'
full.loc[full['label'] == 'land','label'] = 'dos'
full.loc[full['label'] == 'pod','label'] = 'dos'
full.loc[full['label'] == 'smurf','label'] = 'dos'
full.loc[full['label'] == 'teardrop','label'] = 'dos'
full.loc[full['label'] == 'mailbomb','label'] = 'dos'
full.loc[full['label'] == 'apache2','label'] = 'dos'
full.loc[full['label'] == 'udpstorm','label'] = 'dos'
full.loc[full['label'] == 'processtable','label'] = 'dos'
full.loc[full['label'] == 'worm','label'] = 'dos'

full.loc[full['label'] == 'guess_passwd','label'] = 'r2l'
full.loc[full['label'] == 'ftp_write','label'] = 'r2l'
full.loc[full['label'] == 'imap','label'] = 'r2l'
full.loc[full['label'] == 'phf','label'] = 'r2l'
full.loc[full['label'] == 'multihop','label'] = 'r2l'
full.loc[full['label'] == 'warezmaster','label'] = 'r2l'
full.loc[full['label'] == 'warezclient','label'] = 'r2l'
full.loc[full['label'] == 'spy','label'] = 'r2l'
full.loc[full['label'] == 'xlock','label'] = 'r2l'
full.loc[full['label'] == 'xsnoop','label'] = 'r2l'
full.loc[full['label'] == 'snmpguess','label'] = 'r2l'
full.loc[full['label'] == 'snmpgetattack','label'] = 'r2l'
full.loc[full['label'] == 'httptunnel','label'] = 'r2l'
full.loc[full['label'] == 'sendmail','label'] = 'r2l'
full.loc[full['label'] == 'named','label'] = 'r2l'

full.loc[full['label'] == 'satan', 'label'] = 'probe'
full.loc[full['label'] == 'ipsweep', 'label'] = 'probe'
full.loc[full['label'] == 'nmap', 'label'] = 'probe'
full.loc[full['label'] == 'portsweep', 'label'] = 'probe'
full.loc[full['label'] == 'saint', 'label'] = 'probe'
full.loc[full['label'] == 'mscan', 'label'] = 'probe'

#assert that there are no missing data
print('Any missing data on the dataset ???')
print(full.isnull().values.any())
 
#Handling categorical data
full2 = pd.get_dummies(full, drop_first=False)

#Feature squeezing
'''
full2 = full2. drop(columns= [
                'srv_count', 
                'count', 
                'dst_host_same_src_port_rate', 
                'dst_host_srv_count', 
                'dst_bytes', 
                'src_bytes', 
                'dst_host_diff_srv_rate', 
                'dst_host_rerror_rate', 
                'same_srv_rate',
                'dst_host_same_srv_rate']) 
'''

#returns the column labels of the dataframe
features = list(full2.columns[:-5]) #last five columns correspond to labels
 
X_train = full2[0:df.shape[0]] [features]
Y_train = np.array(full2[0:df.shape[0]][['label_dos','label_normal','label_probe','label_r2l','label_u2r']])
 
X_test = full2[df.shape[0]:][features]
Y_test = np.array(full2[df.shape[0]:][['label_dos','label_normal','label_probe','label_r2l','label_u2r']])
 
 #feature scaling
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = np.array(scaler.transform(X_train))
X_test_scaled = np.array(scaler.transform(X_test))

print("------ END OF DATA PREPROCESSING --------")

###########################################################################
    # END OF DATA PREPROCESSING
###########################################################################










###########################################################################
                    # RESULTS ON CLEAN DATA
###########################################################################
#reshaping inputs to be [samples, time steps, features]
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0],1,X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0],1,X_test_scaled.shape[1]))

def lstm_model():
        #Build the model
        lstm = Sequential()
        lstm.add(LSTM(units = 5, input_shape = (None, 122), return_sequences=True))
        lstm.add(Dropout(0.1))
        
        #3rd LSTM layer
        #lstm.add(LSTM(units = 5, return_sequences=True))
        #lstm.add(Dropout(0.1))
        
        #lstm.add(LSTM(units = 5, return_sequences=True))
        #lstm.add(Dropout(0.1))
         
        #2nd LSTM layer
        lstm.add(LSTM(units = 5, return_sequences=False))
        lstm.add(Dropout(0.1))          

        lstm.add(Dense(output_dim = 5)) 
        lstm.add(Activation ('softmax')) #TUNING

        lstm.compile(optimizer = 'adam', 
                     loss = 'categorical_crossentropy', 
                     metrics=['accuracy'])        
        return lstm

    
def evaluate():
        # Evaluate the accuracy of the model on legitimate test examples
        eval_params = {'batch_size' : 128}
        accuracy = model_eval(sess,x,y,predictions, X_test_scaled, Y_test, args = eval_params)
        print ('Test accuracy on legitimate test examples : ' + str(accuracy))
        

# Create TF session and 
sess = tf.Session()
print("Created TensorFlow session.")

#set as Keras backend session
keras.backend.set_session(sess)

#Define input tensorflow placeholders
x = tf.placeholder(tf.float32,shape = (None, 1, X_train_scaled.shape[2]))
y = tf.placeholder(tf.float32,shape = (None, Y_train.shape[1]))

#define the model 
model = lstm_model() 
predictions = model(x) #placeholder, not original data
init = tf.global_variables_initializer()
sess.run(init)


#Train the model
train_params = {
                'nb_epochs': 50, 
                'batch_size' : 128,
                'learning_rate' : 0.01, #TODO TUNING
                'verbose' : 0   
                }

rng = np.random.RandomState([2018, 9, 26])

start_time = time.time(); #measure training time

#train the model, not test
model_train(sess, 
            x, 
            y, 
            predictions, 
            X_train_scaled, 
            Y_train, 
            evaluate=evaluate,
            args = train_params)

print("------ EVALUATION OF MLP performance ------------------")
print()


#Evaluate the accuracy of the model on legitimate test examples
eval_params = {'batch_size' : 128}
accuracyNormal = model_eval(sess, x, y, predictions, X_test_scaled, Y_test, args = eval_params)
print('Test accuracy on normal examples ' + str(accuracyNormal))

elapsed_time = time.time() - start_time
print('Elapsed time on training : %0.4f\n' % elapsed_time)







###########################################################################
    # Craft adversarial examples using FGSM approach
    #cleverhans/cleverhans_tutorials/mnist_tutorial_keras_tf.py
    
        #adversarial test set on X_test_adv
###########################################################################
    
# Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
wrap = KerasModelWrapper(model)

start_time_adv = time.time()

fgsm = FastGradientMethod(wrap, sess=sess)

fgsm_params = {        
           'eps': 0.03, #TODO TUNING
           'clip_min': 0.,
           'clip_max': 1.
           }

adv_x = fgsm.generate(x, **fgsm_params)

# Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)
preds_adv = model(adv_x)

#evaluate the accuracy on ADVERSARIAL EXAMPLES
eval_par = {'batch_size' : 128}

X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test_scaled])

adv_accuracy = model_eval(sess, x, y , predictions, X_test_adv, Y_test, args = eval_par)

elapsed_time_adv = time.time() - start_time_adv
print('Test accuracy on adversarial examples: %0.4f\n' % adv_accuracy)
print('Elapsed time on FGSM : %0.4f\n' % elapsed_time_adv)





###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
                    #adversarial test set on adv_x
###########################################################################
nb_classes = Y_train.shape[1]   
source_samples = X_test_scaled.shape[0]

nb_adv_per_sample = 1

print('Crafting ' + str(source_samples) + ' * ' + str(nb_adv_per_sample) +
          ' adversarial examples')
print("This could take some time ...")

start_time_adv = time.time()
  
# Instantiate a CW attack object
cw = CarliniWagnerL2(model, back='tf', sess=sess)

adv_inputs = X_test_scaled[:source_samples]
 
adv_ys = None
yname = "y"
        
cw_params = {'binary_search_steps': 1,
                  yname: adv_ys,
                  'confidence': 0,
                 'max_iterations': 100,
                 'learning_rate': .001,
                 'batch_size': source_samples,
                 'initial_const': 10,
                 'clip_min' : 0.0,
                 'clip_max' : 1.0,
                 }

adv_x = cw.generate_np(adv_inputs, **cw_params)

eval_params = {'batch_size': np.minimum(nb_classes, source_samples)}

preds = model(x)
adv_accuracy = 1 - \
                model_eval(sess, x, y, preds, adv_x, Y_test[:source_samples], args=eval_params)
        
elapsed_time_adv = time.time() - start_time_adv

print('Test accuracy on C&W examples: %0.4f\n' % adv_accuracy)
print('Elapsed time on C&W : %0.4f\n' % elapsed_time_adv)   
        

    

###########################################################################
    # Craft adversarial examples using Deepfool approach
    
    #adversarial test set on X_test_adv
###########################################################################

wrap = KerasModelWrapper(model)

start_time_adv = time.time()

deepFool = DeepFool(wrap, sess=sess)

deepFool_params = {'nb_candidate': 4,
   'overshoot': 0.05, #TODO TUNING
   'max_iter': 50,
   'nb_classes' : 5,
   'clip_min' : 0.0,
   'clip_max' : 1.0
   }

adv_x = deepFool.generate(x, **deepFool_params)

# Consider the attack to be constant
adv_x = tf.stop_gradient(adv_x)

preds_adv = model(adv_x)

#evaluate the accurayc on ADVERSARIAL EXAMPLES
eval_par = {'batch_size' : 128}

X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test_scaled])


adv_accuracy = model_eval(sess, x, y , predictions, X_test_adv, Y_test, args = eval_par)
elapsed_time_adv = time.time() - start_time_adv

print('Test accuracy on adversarial examples: %0.4f\n' % adv_accuracy)
print('Elapsed time on DeepFool : %0.4f\n' % elapsed_time_adv)
   
  
    


###########################################################################
                    # Adversarial feature statistics
###########################################################################
#find drop in accuracy
dropAccuracy = accuracyNormal - adv_accuracy
print('Drop in accuracy : ', dropAccuracy*100)

feats = dict()
total = 0

#orig_attack = X_test_scaled - adv_x #For calini and wagner

orig_attack = X_test_scaled - X_test_adv #for FGSM and DeepFool attacks, why ??

for i in range(0, orig_attack.shape[0]):
        ind = np.where(orig_attack[i,0,:] !=0)[0] #store indexes for changed feature
        #print(ind)
        total+= len(ind)
        
        for j in ind:
                if j in feats:
                        feats[j] += 1
                else:
                        feats[j] = 1
                        
#the number of features that where changed for the adversarial  samples
print('Number of unique features changed : ', len(feats.keys()))
print('Number of average features changed per datapoint ', total/len(orig_attack))
        
#special case for finding size of tensor object
print('Number of average features changed per datapoint ', total/22543)

print('Ultimate parameter : ', (dropAccuracy*100) / (total/22543))

#store indexes of most changed features
top_10 = sorted(feats, key=feats.get, reverse = True)[:10]
top_20 = sorted(feats, key=feats.get, reverse = True)[:20]

#store percentage of most change features
top_10_val = [100*feats[k] / Y_test.shape[0] for k in top_10]
top_20_val = [100*feats[k] / Y_test.shape[0] for k in top_20]

#plot features in descending order
plt.figure(figsize = (16,12))
plt.bar(np.arange(20), top_20_val, align = 'center')
plt.xticks(np.arange(20), X_train.columns[top_20], rotation='vertical')
plt.ylabel('Percentage (%)')
plt.xlabel('Features')
plt.savefig('lstmc&w.png')