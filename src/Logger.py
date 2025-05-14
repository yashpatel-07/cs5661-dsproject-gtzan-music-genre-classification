from itertools import cycle
import pathlib
import keras
import pandas
import tensorflow as tf
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import io


colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkviolet", "crimson", "lime", "yellow", "black", "red", "blue"])

class TensorBoard(keras.callbacks.TensorBoard):
    '''
    Custom TensorBoard callback to log confusion matrix images.
    '''

    def __init__(self, validation_data, log_dir, batch_size=32, categories=None, sample_rate=22500, *args, **kwargs):
        super().__init__(log_dir=log_dir, *args, **kwargs)
        self.validation_data = validation_data
        self.log_dir = str(log_dir)
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        
        if categories is not None:
            self.categories = categories.categories.to_numpy()

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        file_writer = tf.summary.create_file_writer(str(self.log_dir))
        val_images, val_labels = self.validation_data
        predictions = self.model.predict(
            val_images, batch_size=self.batch_size)
        best_predictions = np.argmax(predictions, axis=1)

        self.confusion_matrix(val_labels, best_predictions, file_writer, epoch)
        self.roc_curve(val_labels, best_predictions, file_writer, epoch)
        self.misclassified_audio(
            val_labels, best_predictions, file_writer, epoch)
        
        with file_writer.as_default():
            precision, recall, fbeta_score, _ = sklearn.metrics.precision_recall_fscore_support(val_labels, best_predictions, average='weighted')
            roc_auc = sklearn.metrics.roc_auc_score(val_labels, predictions, multi_class='ovr', average='weighted')
            
            tf.summary.scalar('Precision', precision, step=epoch)
            tf.summary.scalar('Recall', recall, step=epoch)
            tf.summary.scalar('F1 Score', fbeta_score, step=epoch)
            tf.summary.scalar('ROC AUC', roc_auc, step=epoch)
        file_writer.close()

    def confusion_matrix(self, y_true, y_pred, file_writer, epoch):
        """
        Compute the confusion matrix.
        """

        cm = tf.math.confusion_matrix(y_true, y_pred)

        # Convert confusion matrix to an image
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm.numpy(),
                                                      display_labels=self.categories)
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - Epoch {epoch+1}")

        self.log_plot(fig, file_writer, epoch, "Confusion Matrix") # Log the confusion matrix to TensorBoard
        
        
    

    def roc_curve(self, y_true, y_pred, file_writer, epoch):
        """
        Compute the ROC curve.
        """

        # Convert confusion matrix to an image
        fig, ax = plt.subplots(figsize=(10, 10))
        for i, name in enumerate(self.categories):
            sklearn.metrics.RocCurveDisplay.from_predictions(
                y_true == i,
                y_pred == i,
                # despine=True,
                ax=ax,
                color="blue",
                name=f"ROC curve for {name}",
                plot_chance_level=i == 0,
            )
        plt.title(f"ROC Curve - Epoch {epoch+1}")

        self.log_plot(fig, file_writer, epoch, "ROC curve") # Log the ROC curve matrix to TensorBoard
        

        
    def log_plot(self, fig, file_writer, epoch, name):
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        # Log the image to TensorBoard
        with file_writer.as_default():
            self.summary.image(name, image, step=epoch)
        print(self.log_dir)
        plt.close(fig)
        buf.close()
    
    def misclassified_audio(self, y_true, y_pred, file_writer, epoch):
        """
        Log misclassified audio to TensorBoard.
        """
        misclassified_indices = np.where(y_true != y_pred)[0]

        # Log the audio to TensorBoard
        audio_tensor = self.validation_data[0][misclassified_indices]
        audio_tensor = tf.convert_to_tensor(audio_tensor)
        audio_tensor = tf.expand_dims(audio_tensor, axis=-1)
        with file_writer.as_default():
            tf.summary.audio(f"Misclassified Clips", audio_tensor, sample_rate=self.sample_rate, step=epoch,
                             description='\n'.join([f"True: {y_true[i]}, Pred: {y_pred[i]}" for i, _ in zip(misclassified_indices, range(3))]))
