using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using System.IO;

public class BERTQuestionAnswering : MonoBehaviour
{
    private Model runtimeModel;
    private IWorker worker;
    private TensorInt inputTensor;
    private TensorInt attentionMaskTensor;
    private TensorFloat startLogitsTensor;
    private TensorFloat endLogitsTensor;
    private BERTTokenizer tokenizer;

    void Start()
    {
        // Load ONNX model from Assets/Resources/model-4.onnx
        ModelAsset modelAsset = Resources.Load<ModelAsset>("model-4");
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);

        // Initialize the tokenizer with the vocabulary file
        tokenizer = new BERTTokenizer(Application.dataPath + "/vocab.txt");

        // Example question and context
        string question = "What is the primary function of the heart?";
        string context = "The heart is a muscular organ located in the chest, slightly to the left of the midline. " +
            "It is responsible for pumping blood throughout the body via the circulatory system. " +
            "The heart consists of four chambers: two atria and two ventricles. " +
            "Its primary function is to supply oxygenated blood to the tissues and organs, " +
            "and to return deoxygenated blood to the lungs for reoxygenation. " +
            "This process is essential for maintaining life and supporting the body's metabolic needs.";


        // Tokenize and create input tensors
        CreateInputTensors(question, context);

        // Run the model
        var inputs = new Dictionary<string, Tensor>
        {
            { "input_ids", inputTensor },
            { "attention_mask", attentionMaskTensor }
        };
        worker.Execute(inputs);

        // Get logits tensors
        startLogitsTensor = worker.PeekOutput("start_logits") as TensorFloat;
        endLogitsTensor = worker.PeekOutput("end_logits") as TensorFloat;
        startLogitsTensor.MakeReadable();
        endLogitsTensor.MakeReadable();

        // Print logits tensors shapes and data
        PrintLogitsTensor(startLogitsTensor, "Start Logits");
        PrintLogitsTensor(endLogitsTensor, "End Logits");

        // Make the input tensor readable
        inputTensor.MakeReadable();

        // Find the answer span
        string answer = GetAnswerFromLogits(startLogitsTensor, endLogitsTensor, inputTensor.ToReadOnlyArray(), tokenizer);
        Debug.Log("Answer: " + answer);

        // Clean up
        OnDestroy();
    }

    void CreateInputTensors(string question, string context)
    {
        string inputText = $"[CLS] {question} [SEP] {context} [SEP]";
        int[] inputIds = tokenizer.Tokenize(inputText);
        int[] attentionMask = new int[inputIds.Length];
        for (int i = 0; i < inputIds.Length; i++)
        {
            attentionMask[i] = 1; // Mask all tokens
        }

        inputTensor = new TensorInt(new TensorShape(1, inputIds.Length), inputIds);
        attentionMaskTensor = new TensorInt(new TensorShape(1, attentionMask.Length), attentionMask);
    }

    void PrintLogitsTensor(TensorFloat logits, string name)
    {
        Debug.Log($"{name} Tensor Shape: {string.Join(", ", logits.shape)}");

        for (int i = 0; i < logits.shape[1]; i++)
        {
            Debug.Log($"{name} Token {i}: Logit = {logits[0, i]}");
        }
    }

    string GetAnswerFromLogits(TensorFloat startLogits, TensorFloat endLogits, int[] inputIds, BERTTokenizer tokenizer)
    {
        // Find the start and end positions
        int startIndex = 0;
        int endIndex = 0;
        float maxStartLogit = float.NegativeInfinity;
        float maxEndLogit = float.NegativeInfinity;

        for (int i = 0; i < startLogits.shape[1]; i++)
        {
            if (startLogits[0, i] > maxStartLogit)
            {
                maxStartLogit = startLogits[0, i];
                startIndex = i;
            }

            if (endLogits[0, i] > maxEndLogit)
            {
                maxEndLogit = endLogits[0, i];
                endIndex = i;
            }
        }

        // Debug logs to check start and end indices
        Debug.Log($"Start Index: {startIndex}, Start Logit: {maxStartLogit}");
        Debug.Log($"End Index: {endIndex}, End Logit: {maxEndLogit}");

        // Ensure startIndex is before endIndex
        if (startIndex > endIndex)
        {
            int temp = startIndex;
            startIndex = endIndex;
            endIndex = temp;
        }

        // Convert token indices to text
        string[] tokens = tokenizer.Decode(inputIds).Split(' ');
        if (endIndex >= tokens.Length)
        {
            endIndex = tokens.Length - 1;
        }

        // Debug log to check token range
        Debug.Log($"Tokens from {startIndex} to {endIndex}");

        string answer = string.Join(" ", tokens, startIndex, endIndex - startIndex + 1);

        // Remove special tokens from the answer
        answer = answer.Replace("[CLS]", "").Replace("[SEP]", "").Trim();

        return answer;
    }


    void OnDestroy()
    {
        // Clean up
        inputTensor.Dispose();
        attentionMaskTensor.Dispose();
        startLogitsTensor.Dispose();
        endLogitsTensor.Dispose();
        worker.Dispose();
    }
}
