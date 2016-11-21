#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iterator>
#include <set>
#include <map>
#include <math.h> 

using namespace std;

struct biography {
	biography(string inName, string inCategory, vector<string> inWords) :name(inName), category(inCategory), words(inWords.begin(), inWords.end()) {};
	string name;
	string category;
	set<string> words;
};

//given a space delimited string, return a vector of the parts
vector<string> tokenize(string& input) {
	stringstream ss(input);
	istream_iterator<string> begin(ss);
	istream_iterator<string> end;
	vector<string> vstrings(begin, end);
	return vstrings;
}

void removePunc(string& input) {
	input.erase(remove_if(input.begin(), input.end(), ispunct), input.end());
}

bool isShort(string input) { return (input.empty() || input.length() < 3); }

vector<string> normalize(vector<string> input, vector<string> stop_words) {
	vector<string> output;
	for (string word : input) {
		transform(word.begin(), word.end(), word.begin(), ::tolower);
		removePunc(word);
		if (!isShort(word)) {
			output.push_back(word);
		}
	}
	for (string word : stop_words) {
		vector<string>::iterator it = find(output.begin(), output.end(), word);
		if ( it != output.end()) {
			output.erase(it);
		}
	}
	return output;
}

int main(int argc, char* argv[]) {
	if (argc < 3) {
		cout << "Usage: BayesClassifier inputfile.txt N stopwords.txt" << endl;
		exit(EXIT_FAILURE);
	}

	//get stop words first
	vector<string> stop_words;
	ifstream stopFile(argv[3]);
	if (stopFile.is_open())
	{
		string line;
		while (getline(stopFile, line))
		{
			vector<string> subparts = tokenize(line);
			stop_words.insert(stop_words.end(), subparts.begin(), subparts.end());
		}
		stopFile.close();
	}
	else {
		cout << "Unable to open file" << argv[3] << endl;
		exit(EXIT_FAILURE);
	}

	//then read biographies
	ifstream inputFile(argv[1]);
	vector<biography> biographies;
	if (inputFile.is_open())
	{
		string line;
		string name, category;
		vector<string> words;
		while (getline(inputFile, line))
		{
			if (line.empty() || line == " ") {
				vector<string> newWords = normalize(words, stop_words);
				biographies.push_back(biography(name, category, newWords));
				name = "";
				category = "";
				words.clear();
			}
			else {
				if (name.empty() || name == "") {
					name = line;
				}
				else if (category.empty() || category == "") {
					category = line;
				}
				else {
					vector<string> subparts = tokenize(line);
					words.insert(words.end(), subparts.begin(), subparts.end());
				}
			}		
		}
		vector<string> newWords = normalize(words, stop_words);
		biographies.push_back(biography(name, category, newWords));
		inputFile.close();
	}
	else {
		cout << "Unable to open file" << argv[1] << endl;
		exit(EXIT_FAILURE);
	}

	size_t training_set_size;
	istringstream(argv[2]) >> training_set_size;
	
	vector<biography> training_set(biographies.begin(), biographies.begin() + training_set_size);
	vector<biography> test_set((biographies.begin() + training_set_size ), biographies.end());
	map<string, double> category_count;
	map<string, map<string, double>> word_count;

	for (biography currBio : training_set) {
		//this is a new category
		if (category_count.find(currBio.category) == category_count.end()) {
			category_count[currBio.category] = 1;
			word_count[currBio.category];
			for (pair<string, double> words : word_count.begin()->second) {
				word_count[currBio.category][words.first] = 0;
			}
		}
		else {
			category_count[currBio.category] += 1;
		}
		for (string word : currBio.words) {
			//this is a new word
			if (word_count[currBio.category].find(word) == word_count[currBio.category].end()) {
				word_count[currBio.category][word] = 1;
				for (pair<string, double> cat : category_count) {
					if (word_count[cat.first].find(word) == word_count[cat.first].end()) {
						word_count[cat.first][word] = 0;
					}
				}
			}
			else {
				word_count[currBio.category][word] += 1;
			}
		}
	}
	
	//calculate frequencies
	for (map<string, double>::iterator cat_iter = category_count.begin(); cat_iter != category_count.end();  cat_iter++) {
		for (map<string, double>::iterator word_iter = word_count[cat_iter->first].begin(); word_iter != word_count[cat_iter->first].end(); word_iter++) {
			word_iter->second = word_iter->second / cat_iter->second;
		}
		cat_iter->second = cat_iter->second / training_set_size;
	}

	//calculate probabilities
	double correct = .1;
	for (map<string, double>::iterator cat_iter = category_count.begin(); cat_iter != category_count.end(); cat_iter++) {
		for (map<string, double>::iterator word_iter = word_count[cat_iter->first].begin(); word_iter != word_count[cat_iter->first].end(); word_iter++) {
			word_iter->second = -1 * log2((word_iter->second + correct) / (1 + (2 * correct)));
		}
		cat_iter->second = -1 * log2((cat_iter->second + correct) / (1 + (category_count.size() * correct)));
	}


	//now for prediction time
	map<string, map<string, double>> test_words;
	size_t correct_count = 0;
	for (biography bio : test_set) {
		cout << bio.name << endl;
		map<string, double> category_probs;
		for (pair<string, map<string, double>> previous : word_count) {
			double weight_sum = 0;
			for (string word : bio.words) {
				if (previous.second.find(word) != previous.second.end()) {
					weight_sum += previous.second[word];
				}
			}
			category_probs[previous.first] = category_count[previous.first] + weight_sum;
		}

		double min_value = category_probs.begin()->second;
		string prediction = category_probs.begin()->first;
		//predict
		for (pair<string, double> category : category_probs) {
			if (category.second < min_value) {
				min_value = category.second;
				prediction = category.first;
			}
		}

		//recover
		double prob_sum = 0;
		for (map<string, double>::iterator cate_iter = category_probs.begin(); cate_iter != category_probs.end(); cate_iter++) {
			if (cate_iter->second - min_value < 7) {
				prob_sum += pow(2, (min_value - cate_iter->second));
				cate_iter->second = pow(2, (min_value - cate_iter->second));
			}
		}

		cout << "probabilities" << endl;
		for (pair<string, double> category : category_probs) {
			cout << category.first << ":" << to_string(category.second / prob_sum) << endl;
		}

		cout << "prediction: " << prediction << endl;

		if (bio.category == prediction) {
			cout << "prediction is right!" << endl;
			correct_count++;
		}
		else {
			cout << "prediction is wrong!" << endl;
		}
	}
	double accuracy = double(correct_count) / double(test_set.size());
	cout << "Accuracy: " << to_string(accuracy) << endl;
}