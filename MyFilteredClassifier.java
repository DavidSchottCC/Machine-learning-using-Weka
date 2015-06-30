package de.codecentric.docnitio.experimenter.helloworld.helloweka;

/**
 * A Java class that implements a simple text classifier, based on WEKA.
 * To be used with MyFilteredLearner.java.
 * WEKA is available at: http://www.cs.waikato.ac.nz/ml/weka/
 */

import weka.core.*;
import weka.classifiers.meta.FilteredClassifier;

import java.util.List;
import java.util.ArrayList;
import java.io.*;

/**
 * This class implements a simple text classifier in Java using WEKA. It loads a
 * file with the text to classify, and the model that has been learnt with
 * MyFilteredLearner.java.
 * 
 * @see MyFilteredLearner
 */
public class MyFilteredClassifier {

	/**
	 * String that stores the text to classify
	 */
	String text;
	/**
	 * Object that stores the instance.
	 */
	Instances instances;
	/**
	 * Object that stores the classifier.
	 */
	FilteredClassifier classifier;

	/**
	 * This method loads the text to be classified.
	 * 
	 * @param filePath
	 *            The name of the file that stores the text.
	 */
	public void loadData(String filePath) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(filePath));
			String line;
			text = "";
			while ((line = reader.readLine()) != null) {
				text = text + " " + line;
			}
			System.out
					.println("===== Loaded text data: " + filePath + " =====");
			reader.close();
			System.out.println(text);
		} catch (IOException e) {
			System.out.println("Problem found when reading: " + filePath);
		}
	}

	/**
	 * This method loads the model to be used as classifier.
	 * 
	 * @param modelPath
	 *            The path of the file that stores the classifier.
	 */
	public void loadModel(String modelPath) {
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					modelPath));
			Object tmp = in.readObject();
			classifier = (FilteredClassifier) tmp;
			in.close();
			System.out.println("===== Loaded model: " + modelPath + " =====");
		} catch (Exception e) {
			// Given the cast, a ClassNotFoundException must be caught along
			// with the IOException
			System.out.println("Problem found when reading: " + modelPath);
		}
	}

	/**
	 * This method creates the instance to be classified, from the text that has
	 * been read.
	 */
	@SuppressWarnings("deprecation")
	public void makeInstances() {
		// Create the 2 attributes, class = {spam, ham} and text = string
		FastVector<String> fvNominalVal = new FastVector<String>(2);
		fvNominalVal.addElement("spam");
		fvNominalVal.addElement("ham");
		Attribute attribute1 = new Attribute("class", fvNominalVal); // @attribute spamclass {spam,ham}
		Attribute attribute2 = new Attribute("text", (FastVector<String>) null); // @attribute text String
		// Create list of instances with capacity 1
		FastVector<Attribute> fvWekaAttributes = new FastVector<Attribute>(2);
		fvWekaAttributes.addElement(attribute1);
		fvWekaAttributes.addElement(attribute2);
		instances = new Instances("Test relation", fvWekaAttributes, 1);
		// Set class index
		instances.setClassIndex(0);
		// Create and add the instance to instances
		DenseInstance instance = new DenseInstance(2);
		instance.setValue((Attribute) fvWekaAttributes.elementAt(1), text);
		instances.add(instance);
		System.out
				.println("===== Instance created with reference dataset =====");
		System.out.println(instances);
	}

	/**
	 * This method performs the classification of the instance. Output is done
	 * at the command-line.
	 */
	public void classify() {
		try {
			double pred = classifier.classifyInstance(instances.instance(0));
			double[] preds = classifier.distributionForInstance(instances
					.instance(0));
			System.out.println("===== Classified instance =====");
			System.out.println("Class predicted: "
					+ instances.classAttribute().value((int) pred));
			System.out.println("Ham probability: " + preds[1]);
			System.out.println("Spam probability: " + preds[0]);
		} catch (Exception e) {
			System.out.println("Problem found when classifying the text");
		}
	}

	/**
	 * Main method. It is an example of the usage of this class.
	 * 
	 * @param args
	 *            Command-line arguments: fileData and fileModel.
	 */
	public static void main(String[] args) {
		String dirPath = "C:\\Users\\DavidCC\\Desktop\\Weka Beispiel\\tmweka\\FilteredClassifier\\";
		MyFilteredClassifier classifier;
		classifier = new MyFilteredClassifier();
		classifier.loadData(dirPath + "smstest.txt");
		classifier.loadModel(dirPath + "trained-classifier.obj");
		classifier.makeInstances();
		classifier.classify();
	}
}