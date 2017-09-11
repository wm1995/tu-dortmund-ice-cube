from myTools.waveform_tools.waveform_generator import WaveformGenerator

from keras.callbacks import TensorBoard

def train_model(model, data, params, nn_str, comment="", verbose=False, initial_epoch=0):
    # Create generators for training, validation
    train_gen = WaveformGenerator(
            data.train, 
            batch_size=params['batch_size'], 
            balanced=True, 
            dp_prob=params['dp_prob'],
            decay=params['decay']
        )

    val_gen = WaveformGenerator(
            data.val, 
            batch_size=params['batch_size'], 
            balanced=True, 
            dp_prob=params['dp_prob'],
            decay=params['decay']
        )

    # Prepare callbacks
    callbacks = [train_gen, val_gen]

    if test == False:
        model_saver = ModelSaver(
                model, 'retrain', params, 
                comment=comment,
                verbose=verbose, period=cp_interval
            )
        tb = TensorBoard(log_dir='logs/logs-' + model_saver.model_name)
        callbacks += [tb, model_saver]

    # Train model
    model.fit_generator(
            train_gen, 
            steps_per_epoch=params['steps_per_epoch'], 
            epochs=params['no_epochs'],
            verbose=int(verbose), 
            validation_data=val_gen,
            validation_steps=params['steps_per_epoch'], 
            callbacks=callbacks,
            initial_epoch=initial_epoch
        )