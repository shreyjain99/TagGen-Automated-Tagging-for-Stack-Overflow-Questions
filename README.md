
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
<p><b><u> Time-series forecasting and Regression</u></b></p>
-<i> To find number of pickups, given location cordinates(latitude and longitude) and time, in the query reigion and surrounding regions therefore we predict number of pickups accurately as possible for each region in a 10 min interval.</i> (we would be using data collected in Jan - Mar 2015 to predict the pickups in Jan - Mar 2016.)

<br>
<br>

<p>
<strong>Performance metrics :</strong>
</p>
<ol>
<li>Mean Absolute percentage error</li>
<li>Mean Squared error</li>
</ol>

<hr width="100%" size="2">

<br>

<body>

  <h3>Summary</h3>

  <h4>Data Processing</h4>
    <p>The project begins with data cleaning and the removal of outliers through univariate analysis of key features. This step ensures that the data used for modeling is both accurate and relevant.</p>

  <h4>Clustering and Region Division</h4>
    <p>Following data cleaning, the city is divided into 40 regions based on clustering techniques that consider both distance and time intervals. This division allows for a more granular and accurate prediction of taxi demand across the city.</p>

  <h4>Prediction Framework</h4>
    <p>For each region, the data is broken down into 10-minute time intervals, and the number of pickups within each interval is predicted. The modeling process begins with baseline models, including moving averages, moving weighted averages, and exponential averages, which serve as simple predictors of demand.</p>

   <h4>Advanced Modeling</h4>
    <p>The project then advances to more sophisticated machine learning models, including linear regression, Random Forest regressor, and XGBoost regressor. These models are employed to enhance prediction accuracy by leveraging the complex relationships in the data.</p>

  <p>The final output of the project is a robust predictive model that can help taxi drivers optimize their routes and improve their earnings by forecasting the most promising areas for pickups in real-time.</p>

</body>

<hr width="100%" size="2">
<br>

<p>
<strong>Future Scope :</strong>
</p>
<ol>
<li>Incorporate fourier transform features as features into regression models </li>
<li>Perform hyperparameter tuning for regression models</li>
<li>Try more regression models as well as artificial neural nets (ann)</li>
</ol>

<hr width="100%" size="2">
