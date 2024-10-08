
<h2 align= "center"><em>TagGen: Automated Tagging for Stack Overflow Questions</em></h2>

<div align="center">
  <img height="400" src="https://github.com/shreyjain99/TagGen-Automated-Tagging-for-Stack-Overflow-Questions/blob/main/src%20files/pic1.jpg"/>
</div>

<hr width="100%" size="2">

<h3 align= "left"> <b> Key Project Formulation </b> </h3>

<br>

<p>
<strong>Real World/Business Objective :</strong> Incorrect tags could impact customer experience on StackOverflow therefore Predict as many tags as possible with high precision and recall.
</p>

<br>

<p>
<strong>Constraints :</strong>
</p>
<ol>
<li>High precision and recall </li>
<li>No latency constraints</li>
</ol>

<br>

<p>
<strong>Get the data from :</strong> https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data
<br>The data is from facebook recruiting prediction competition hosted on kaggle.
</p>

<br>

<p>
<strong>Data Collection :</strong>
<br>
All of the data is in 2 files: Train and Test.<br />
<pre>
<b>Train.csv</b> contains 4 columns: Id,Title,Body,Tags.<br />
<b>Test.csv</b> contains the same columns but without the Tags, which you are to predict.<br />
<b>Size of Train.csv</b> - 6.75GB<br />
<b>Size of Test.csv</b> - 2GB<br />
<b>Number of rows in Train.csv</b> = 6034195<br />
</pre>
The questions are randomized and contains a mix of verbose text sites as well as sites related to math and programming. The number of questions from each site may vary, and no filtering has been performed on the questions (such as closed questions).<br />
<br />

<br>

<p>
<strong>Features in the dataset :</strong>
</p>
Dataset contains 6,034,195 rows. The columns in the table are:<br />
<pre>
<b>Id</b> - Unique identifier for each question<br />
<b>Title</b> - The question's title<br />
<b>Body</b> - The body of the question<br />
<b>Tags</b> - The tags associated with the question in a space-seperated format (all lowercase, should not contain tabs '\t' or ampersands '&')<br />
</pre>

<br />

<br>

<p>
<strong>ML Problem Formulation :</strong>
</p>
<p> <strong>It is a multi-label classification problem</strong> </p>
<p> 
<b>Multi-label Classification</b>: Multilabel classification assigns to each sample a set of target labels. This can be thought as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A question on Stackoverflow might be about any of C, Pointers, FileIO and/or memory-management at the same time or none of these. <br>
Credit: http://scikit-learn.org/stable/modules/multiclass.html
</p>

<br>
<br>

<p>
<strong>Performance metrics :</strong>
</p>
<ol>
<li>Micro-Averaged F1-Score (Mean F Score) </li>
<li>Hamming loss</li>
</ol>

<hr width="100%" size="2">

<br>

<body>

  <h3>Flow of Project : </h3>
  
  <br>

  <h3 align= "center"><strong>Data Loading</strong></h3>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center"><strong>Analysis of tags</strong></h3>
  <p align= "center"><em> - Total number of unique tags </em></p>
  <p align= "center"><em> - Number of times a tag appeared </em></p>
  <p align= "center"><em> - Tags Per Question </em></p>
  <p align= "center"><em> - Most Frequent Tags </em></p>
  <p align= "center"><em> - The top 20 tags </em></p>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Data Preprocessing (Natural Language Processing)</h3>
  <p align= "center"><em> - Separated out code-snippets from Body </em></p>
  <p align= "center"><em> - Removed Special characters from Question title and description </em></p>
  <p align= "center"><em> - Removed stop words (Except 'C') </em></p>
  <p align= "center"><em> - Converted all the characters into small letters </em></p>
  <p align= "center"><em> - Used SnowballStemmer to stem the words </em></p>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Machine Learning Models</h3>
  <p align= "center"><em> - Converted tags for multilabel problems </em></p>
  <p align= "center"><em>- Splited the data into test and train (80:20)  </em></p>
  <p align= "center"><em> - Featurized data (TFIDF FEATURES) </em></p>
  <p align= "center"><em> - Applyied Logistic Regression with OneVsRest Classifier</em></p>


  
</body>

<hr width="100%" size="2">
<br>

<div align="center">
  <img height="400" src="https://github.com/shreyjain99/TagGen-Automated-Tagging-for-Stack-Overflow-Questions/blob/main/src%20files/frequent%20tags.png"/>
</div>

<p>
<strong>Future Scope :</strong>
</p>
<ol>
<li>Use bag of words upto 4 grams and compute the micro f1 score with Logistic regression(OnevsRest) </li>
<li>Perform hyperparameter tuning on alpha (or lambda) for Logistic regression to improve the performance using GridSearch</li>
<li>Try OneVsRestClassifier with Linear-SVM (SGDClassifier with loss-hinge)</li>
</ol>

<hr width="100%" size="2">
<br>

<p>
<strong>Skills and Software Tools Used in the Project :</strong>
</p>
<ul>
    <li>Data Analysis</li>
    <li>Natural Language Processing (NLP)</li>
    <li>Multi-label Classification</li>
    <li>Text Preprocessing</li>
    <li>Feature Engineering (TF-IDF)</li>
    <li>Logistic Regression</li>
    <li>OneVsRest Classifier</li>
    <li>Python</li>
    <li>Pandas</li>
    <li>NumPy</li>
    <li>Scikit-learn</li>
</ul>

<hr width="100%" size="2">
