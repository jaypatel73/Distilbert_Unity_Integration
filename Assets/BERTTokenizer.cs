using System;
using System.Collections.Generic;
using System.IO;

public class BERTTokenizer
{
    private Dictionary<string, int> tokenDict;
    private Dictionary<int, string> idToTokenDict;
    private const string unkToken = "[UNK]";
    private const string clsToken = "[CLS]";
    private const string sepToken = "[SEP]";

    public BERTTokenizer(string vocabFilePath)
    {
        tokenDict = new Dictionary<string, int>();
        idToTokenDict = new Dictionary<int, string>();
        LoadVocab(vocabFilePath);
    }

    private void LoadVocab(string vocabFilePath)
    {
        string[] lines = File.ReadAllLines(vocabFilePath);
        for (int i = 0; i < lines.Length; i++)
        {
            string token = lines[i].Trim();
            tokenDict[token] = i;
            idToTokenDict[i] = token;
        }
    }

    public int[] Tokenize(string text)
    {
        List<int> tokenIds = new List<int>();
        tokenIds.Add(tokenDict[clsToken]);

        string[] words = text.Split(' ');
        foreach (var word in words)
        {
            if (tokenDict.ContainsKey(word))
            {
                tokenIds.Add(tokenDict[word]);
            }
            else
            {
                // WordPiece tokenization
                List<string> subWords = WordPieceTokenize(word);
                foreach (var subWord in subWords)
                {
                    tokenIds.Add(tokenDict.ContainsKey(subWord) ? tokenDict[subWord] : tokenDict[unkToken]);
                }
            }
        }

        tokenIds.Add(tokenDict[sepToken]);
        return tokenIds.ToArray();
    }

    private List<string> WordPieceTokenize(string word)
    {
        List<string> subWords = new List<string>();
        string currentSubWord = "";
        for (int i = 0; i < word.Length; i++)
        {
            currentSubWord += word[i];
            if (tokenDict.ContainsKey(currentSubWord))
            {
                subWords.Add(currentSubWord);
                currentSubWord = "";
            }
        }
        if (!string.IsNullOrEmpty(currentSubWord))
        {
            subWords.Add(unkToken);
        }
        return subWords;
    }

    public string Decode(int[] tokenIds)
    {
        List<string> tokens = new List<string>();
        foreach (var tokenId in tokenIds)
        {
            if (idToTokenDict.TryGetValue(tokenId, out string token))
            {
                tokens.Add(token);
            }
            else
            {
                tokens.Add(unkToken);
            }
        }
        return string.Join(" ", tokens);
    }
}
