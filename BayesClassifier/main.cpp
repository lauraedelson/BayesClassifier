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
	string output;
	for (size_t i = 0; i < input.length(); i++) {
		if (!ispunct(input[i])) {
			output.push_back(input[i]);
		}
	}
	input = output;
}

bool isShort(string input) { return (input.empty() || input.length() < 3); }

vector<string> normalize(vector<string> input, vector<string> stop_words) {
	vector<string> output;
	for (vector<string>::iterator in_iter = input.begin(); in_iter != input.end(); in_iter++) {
		transform(in_iter->begin(), in_iter->end(), in_iter->begin(), ::tolower);
		removePunc(*in_iter);
		if (!isShort(*in_iter)) {
			output.push_back(*in_iter);
		}
	}
	for (vector<string>::iterator stop_iter = stop_words.begin(); stop_iter != stop_words.end(); stop_iter++) {
		vector<string>::iterator it = find(output.begin(), output.end(), *stop_iter);
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

	for (vector<biography>::iterator bio_iter = training_set.begin(); bio_iter != training_set.end(); bio_iter++) {
		//this is a new category
		if (category_count.find(bio_iter->category) == category_count.end()) {
			category_count[bio_iter->category] = 1;
			word_count[bio_iter->category];
			for (map<string, double>::iterator words_iter = word_count.begin()->second.begin(); words_iter != word_count.begin()->second.end(); words_iter++) {
				word_count[bio_iter->category][words_iter->first] = 0;
			}
		}
		else {
			category_count[bio_iter->category] += 1;
		}
		for (set<string>::iterator word_iter = bio_iter->words.begin(); word_iter != bio_iter->words.end(); word_iter++) {
			//this is a new word
			if (word_count[bio_iter->category].find(*word_iter) == word_count[bio_iter->category].end()) {
				word_count[bio_iter->category][*word_iter] = 1;
				for (map<string, double>::iterator cat = category_count.begin(); cat != category_count.end(); cat++) {
					if (word_count[cat->first].find(*word_iter) == word_count[cat->first].end()) {
						word_count[cat->first][*word_iter] = 0;
					}
				}
			}
			else {
				word_count[bio_iter->category][*word_iter] += 1;
			}
		}
	}
	
	//calculate frequencies & probabilities
	double correct = .1;
	for (map<string, double>::iterator cat_iter = category_count.begin(); cat_iter != category_count.end();  cat_iter++) {
		for (map<string, double>::iterator word_iter = word_count[cat_iter->first].begin(); word_iter != word_count[cat_iter->first].end(); word_iter++) {
			word_iter->second = word_iter->second / cat_iter->second;
			word_iter->second = -1 * log2((word_iter->second + correct) / (1 + (2 * correct)));
		}
		cat_iter->second = cat_iter->second / training_set_size;
		cat_iter->second = -1 * log2((cat_iter->second + correct) / (1 + (category_count.size() * correct)));
	}

	//now for prediction time (
	map<string, map<string, double>> test_words;
	size_t correct_count = 0;
	for (vector<biography>::iterator bio_iter = test_set.begin(); bio_iter != test_set.end(); bio_iter++) {
		cout << bio_iter->name << endl;
		map<string, double> category_probs;
		for (map<string, map<string, double>>::iterator previous_iter = word_count.begin(); previous_iter != word_count.end(); previous_iter++) {
			double weight_sum = 0;
			for (set<string>::iterator word = bio_iter->words.begin(); word != bio_iter->words.end(); word++) {
				if (previous_iter->second.find(*word) != previous_iter->second.end()) {
					weight_sum += previous_iter->second[*word];
				}
			}
			category_probs[previous_iter->first] = category_count[previous_iter->first] + weight_sum;
		}

		double min_value = category_probs.begin()->second;
		string prediction = category_probs.begin()->first;
		//predict
		for (map<string, double>::iterator cate_iter = category_probs.begin(); cate_iter != category_probs.end(); cate_iter++) {
			if (cate_iter->second < min_value) {
				min_value = cate_iter->second;
				prediction = cate_iter->first;
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
		for (map<string, double>::iterator category = category_probs.begin(); category != category_probs.end(); category++) {
			cout << category->first << ":" << to_string((double long)category->second / prob_sum) << endl;
		}

		cout << "prediction: " << prediction << endl;

		if (bio_iter->category == prediction) {
			cout << "prediction is right!" << endl;
			correct_count++;
		}
		else {
			cout << "prediction is wrong!" << endl;
		}
	}
	double accuracy = double(correct_count) / double(test_set.size());
	cout << "Accuracy: " << to_string((double long)accuracy) << endl;
}
