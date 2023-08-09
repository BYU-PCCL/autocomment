var SEMESTER=0;
var SECTION=1;
var RANDOMID=2;
var COMMENT=3;
var QUESTION=4;
var SCORE=5;
var MAG=6;
var JSCORE=7;
var JMAG=8;
var TOPICX=9;
var TOPICY=10;

function get_full_comments() {
    
    // Create the data table.
    
    // ['F 2016', 'C S 501R (001)', 'Spiritually strengthening', 'Best ever!', score, mag ]
    
    var pos_data = new google.visualization.DataTable();
    pos_data.addColumn('string', 'Semester');
    pos_data.addColumn('string', 'Section');
    pos_data.addColumn('string', 'CommenterID');
    pos_data.addColumn('string', 'Comment');
    pos_data.addColumn('string', 'Question');
    pos_data.addColumn('number', 'Sentiment');
    pos_data.addColumn('number', 'Magnitude');
    pos_data.addColumn('number', 'jitter_Sentiment');
    pos_data.addColumn('number', 'jitter_Magnitude');
    pos_data.addColumn('number', 'topic_x');
    pos_data.addColumn('number', 'topic_y');
    pos_data.addRows([
	{% for comment in full_comments %}
	['{{ comment[0] }}', '{{comment[1]}}', '{{comment[2]}}', '{{comment[3]}}', '{{comment[4]}}', {{comment[5]}}, {{comment[6]}}, {{comment[7]}}, {{comment[8]}}, {{comment[9]}}, {{comment[10]}} ],
	{% endfor %}      
    ]);
    
    return pos_data;
}
