using System.IO;
using UnityEngine;

public class TokenizerTest : MonoBehaviour
{
    private BERTTokenizer tokenizer;

    void Start()
    {
        // Path to your vocabulary file
        string vocabFilePath = Path.Combine(Application.dataPath, "vocab.txt");

        // Initialize the tokenizer
        tokenizer = new BERTTokenizer(vocabFilePath);

        // Test tokenization
        string text = "hello world";
        int[] tokenIds = tokenizer.Tokenize(text);
        Debug.Log("Token IDs: " + string.Join(", ", tokenIds));

        // Test decoding
        string decodedText = tokenizer.Decode(tokenIds);
        Debug.Log("Decoded Text: " + decodedText);
    }
}
