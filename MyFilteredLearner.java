package de.codecentric.docnitio.experimenter.helloworld.helloweka;

/**
 * A Java class that implements a simple text learner, based on WEKA.
 * To be used with MyFilteredClassifier.java.
 * WEKA is available at: http://www.cs.waikato.ac.nz/ml/weka/
 */

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Evaluation;

import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;

import java.io.*;

/**
 * This class implements a simple text learner in Java using WEKA. It loads a
 * text dataset written in ARFF format, evaluates a classifier on it, and saves
 * the learnt model for further use.
 * @see MyFilteredClassifier
 */
public class MyFilteredLearner {

	/**
	 * Object that stores training data.
	 */
	Instances trainData;
	/**
	 * Object that stores the filter
	 */
	StringToWordVector filter;
	/**
	 * Object that stores the classifier
	 */
	FilteredClassifier classifier;

	/**
	 * This method loads a dataset in ARFF format. If the file does not exist,
	 * or it has a wrong format, the attribute trainData is null.
	 * 
	 * @param fileName
	 *            The name of the file that stores the dataset.
	 */
	public void load(String fileName) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(fileName));
			ArffReader arff = new ArffReader(reader);
			trainData = arff.getData();
			System.out.println("===== Loaded dataset: " + fileName + " =====");
			reader.close();
		} catch (IOException e) {
			System.out.println("Problem found when reading: " + fileName);
		}
	}

	/**
	 * This method sets the pre-processing filters to use for preparing the
	 * dataset.
	 */
	public void preProcess() {
		try {
			filter = new StringToWordVector();
			filter.setInputFormat(trainData);
			filter.setAttributeIndices("last");
			//filter.setOutputWordCounts(true);
			// filter.setUseStoplist(true);
			// filter.setIDFTransform(true);
			System.out.println("===== Filter of (training) dataset set =====");
		} catch (Exception e) {
			System.out.println("Problem found when pre-processing data "
					+ e.getMessage());
		}
	}

	/**
	 * This method evaluates the classifier. As recommended by WEKA
	 * documentation, the classifier is defined but not trained yet. Evaluation
	 * of previously trained classifiers can lead to unexpected results.
	 */
	public void evaluate() {
		try {
			// Our training data is tuple of form (class, text)
			trainData.setClassIndex(0); // Class of dataset is first attribute
			// Classifier
			classifier = new FilteredClassifier();
			classifier.setFilter(filter);
			classifier.setClassifier(new NaiveBayes());
			Evaluation eval = new Evaluation(trainData);
			eval.crossValidateModel(classifier, trainData, 10, new Random(1));
			System.out.println(eval.toSummaryString());
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString());
			System.out.println("===== Evaluating on filtered (training) dataset done =====");
		} catch (Exception e) {
			System.out.println("Problem found when evaluating:\t"
					+ e.getMessage());
		}
	}

	/**
	 * This method trains the classifier on the loaded dataset.
	 */
	public void learn() {
		try {
			trainData.setClassIndex(0);
			classifier = new FilteredClassifier();
			classifier.setFilter(filter);
			classifier.setClassifier(new NaiveBayes());
			classifier.buildClassifier(trainData);
			// System.out.println(classifier);
			System.out.println("===== Training on filtered (training) dataset done =====");
		} catch (Exception e) {
			System.out.println("Problem found when training:\t"
					+ e.getMessage());
		}
	}

	/**
	 * This method saves the trained model into a file. This is done by simple
	 * serialization of the classifier object.
	 * 
	 * @param filePath The path of the file that will store the trained model.
	 *            
	 */
	public void saveModel(String filePath) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(
					new FileOutputStream(filePath));
			out.writeObject(classifier);
			out.close();
			System.out.println("===== Saved model: " + filePath + " =====");
		} catch (IOException e) {
			System.out.println("Problem found when writing: " + filePath);
		}
	}

	/**
	 * Applies the preprocessing filters to the dataset and writes an ARFF-file
	 * to a given path.
	 * 
	 * @param savePath
	 *            the full path of the ARFF-file
	 */
	public void saveFilteredData(String savePath) {
		try {
			Instances trainDataFiltered = Filter.useFilter(trainData, filter);
			// if savePath has been set
			if (!savePath.isEmpty()) {
				ArffSaver saver = new ArffSaver();
				saver.setInstances(trainDataFiltered);
				saver.setFile(new File(savePath));
				saver.writeBatch();
				System.out.println("===== Saved filtered (training) dataset to: "
								+ savePath + " =====");
			}
		} catch (Exception e) {
			System.out.println("saving filtered data to Path:\n" + savePath
					+ "\n" + e.getMessage());
		}
	}

	/**
	 * Main method. It is an example of the usage of this class.
	 * 
	 * @param args
	 *            Command-line arguments: fileData-Input and fileModel-Output
	 *            object.
	 */
	public static void main(String[] args) throws Exception {
		MyFilteredLearner learner;
		String datasetPath = "Users/user1/Documents/learner/";
		learner = new MyFilteredLearner();
		learner.load(datasetPath + "smsspam.big.arff");
		learner.preProcess();
		learner.evaluate();
		learner.learn();
		learner.saveModel(datasetPath + "trained-classifier.obj");
	}
}