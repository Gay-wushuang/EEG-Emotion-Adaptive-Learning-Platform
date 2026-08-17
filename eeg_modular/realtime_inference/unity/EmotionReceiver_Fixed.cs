using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using WebSocketSharp;

public class EmotionReceiver : MonoBehaviour
{
    [Header("WebSocket Settings")]
    public string serverUrl = "ws://localhost:8765";
    public float reconnectDelay = 3f;
    
    [Header("Emotion Settings")]
    public float transitionSmoothTime = 1f;
    
    [Header("Skybox Settings")]
    public Material happySkybox;
    public Material sadSkybox;
    public Material normalSkybox;
    
    [Header("Music Settings")]
    public AudioClip happyMusic;
    public AudioClip sadMusic;
    public AudioClip normalMusic;
    public float musicTransitionTime = 2f;
    
    [Header("Debug")]
    public bool showDebugInfo = true;
    public string currentEmotion = "normal";
    public float currentConfidence = 0f;
    public float transitionProgress = 0f;
    
    private WebSocket ws;
    private Coroutine reconnectCoroutine;
    private Coroutine crossfadeCoroutine;
    private string targetEmotion = "normal";
    private float currentMusicVolume = 1f;
    private AudioSource audioSource;
    private AudioSource crossfadeSource;
    private Material currentSkybox;
    private Material targetSkybox;
    private float skyboxBlend = 0f;
    private Dictionary<string, Material> emotionSkyboxes;
    private Dictionary<string, AudioClip> emotionMusic;
    
    private Queue<string> messageQueue = new Queue<string>();
    private object queueLock = new object();
    private bool isDestroying = false;

    private void Start()
    {
        InitializeEmotionMaps();
        InitializeAudio();
        Connect();
    }

    private void InitializeEmotionMaps()
    {
        emotionSkyboxes = new Dictionary<string, Material>
        {
            {"happy", happySkybox},
            {"sad", sadSkybox},
            {"normal", normalSkybox}
        };
        
        emotionMusic = new Dictionary<string, AudioClip>
        {
            {"happy", happyMusic},
            {"sad", sadMusic},
            {"normal", normalMusic}
        };
        
        currentSkybox = normalSkybox;
        targetSkybox = normalSkybox;
        
        if (currentSkybox != null)
        {
            RenderSettings.skybox = currentSkybox;
        }
    }

    private void InitializeAudio()
    {
        audioSource = gameObject.AddComponent<AudioSource>();
        audioSource.loop = true;
        audioSource.volume = 1f;
        audioSource.playOnAwake = false;
        
        crossfadeSource = gameObject.AddComponent<AudioSource>();
        crossfadeSource.loop = true;
        crossfadeSource.volume = 0f;
        crossfadeSource.playOnAwake = false;
        
        if (normalMusic != null)
        {
            audioSource.clip = normalMusic;
            audioSource.Play();
        }
    }

    private void Connect()
    {
        if (!isActiveAndEnabled)
        {
            Debug.LogWarning("[EEG] Cannot connect: GameObject is inactive");
            return;
        }

        if (ws != null && ws.IsAlive)
        {
            return;
        }

        Debug.Log($"[EEG] Connecting to {serverUrl}...");
        ws = new WebSocket(serverUrl);
        
        ws.OnOpen += OnOpen;
        ws.OnMessage += OnMessage;
        ws.OnError += OnError;
        ws.OnClose += OnClose;
        
        ws.ConnectAsync();
    }

    private void OnOpen(object sender, System.EventArgs e)
    {
        Debug.Log("[EEG] ✅ Connected to emotion server!");
        StopReconnectCoroutine();
    }

    private void OnMessage(object sender, MessageEventArgs e)
    {
        lock (queueLock)
        {
            messageQueue.Enqueue(e.Data);
        }
    }

    private void ProcessEmotionData(string jsonData)
    {
        try
        {
            var emotionData = JsonUtility.FromJson<EmotionData>(jsonData);
            
            if (emotionData == null)
            {
                Debug.LogWarning("[EEG] Received null emotion data");
                return;
            }
            
            currentConfidence = emotionData.confidence;
            transitionProgress = emotionData.transition_progress;
            
            if (!string.IsNullOrEmpty(emotionData.emotion) && emotionData.emotion != targetEmotion)
            {
                targetEmotion = emotionData.emotion;
                Debug.Log($"[EEG] 🎭 New emotion detected: {emotionData.emotion} (confidence: {emotionData.confidence:F2})");
                StartEmotionTransition(emotionData.emotion);
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"[EEG] Failed to parse emotion data: {ex.Message}\nRaw data: {jsonData}");
        }
    }

    private void StartEmotionTransition(string newEmotion)
    {
        Debug.Log($"[EEG] Starting emotion transition to: {newEmotion}");
        
        if (emotionSkyboxes.TryGetValue(newEmotion, out Material newSkybox) && newSkybox != null)
        {
            targetSkybox = newSkybox;
            skyboxBlend = 0f;
        }
        
        if (emotionMusic.TryGetValue(newEmotion, out AudioClip newMusic))
        {
            StartCrossfadeCoroutine(newMusic);
        }
    }

    private void StartCrossfadeCoroutine(AudioClip newMusic)
    {
        if (!isActiveAndEnabled)
        {
            Debug.LogWarning("[EEG] Cannot start crossfade: GameObject is inactive");
            return;
        }
        
        if (crossfadeCoroutine != null)
        {
            StopCoroutine(crossfadeCoroutine);
            crossfadeCoroutine = null;
        }
        
        crossfadeCoroutine = StartCoroutine(CrossfadeMusic(newMusic));
    }

    private IEnumerator CrossfadeMusic(AudioClip newClip)
    {
        if (newClip == null) yield break;
        
        crossfadeSource.clip = newClip;
        crossfadeSource.time = audioSource.time;
        crossfadeSource.volume = 0f;
        crossfadeSource.Play();
        
        float elapsed = 0f;
        while (elapsed < musicTransitionTime && isActiveAndEnabled)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / musicTransitionTime;
            
            audioSource.volume = 1f - t;
            crossfadeSource.volume = t;
            
            yield return null;
        }
        
        var temp = audioSource;
        audioSource = crossfadeSource;
        crossfadeSource = temp;
        
        crossfadeSource.Stop();
        crossfadeSource.volume = 0f;
        audioSource.volume = 1f;
        
        crossfadeCoroutine = null;
    }

    private void Update()
    {
        if (!isActiveAndEnabled) return;
        
        ProcessMessageQueue();
        UpdateSkyboxTransition();
    }

    private void ProcessMessageQueue()
    {
        List<string> messagesToProcess = new List<string>();
        
        lock (queueLock)
        {
            while (messageQueue.Count > 0)
            {
                messagesToProcess.Add(messageQueue.Dequeue());
            }
        }
        
        foreach (string message in messagesToProcess)
        {
            ProcessEmotionData(message);
        }
    }

    private void UpdateSkyboxTransition()
    {
        if (currentSkybox != targetSkybox && currentSkybox != null && targetSkybox != null)
        {
            skyboxBlend += Time.deltaTime / transitionSmoothTime;
            skyboxBlend = Mathf.Clamp01(skyboxBlend);
            
            Material blended = new Material(currentSkybox);
            blended.Lerp(currentSkybox, targetSkybox, skyboxBlend);
            RenderSettings.skybox = blended;
            
            if (skyboxBlend >= 1f)
            {
                currentSkybox = targetSkybox;
                currentEmotion = targetEmotion;
                Debug.Log($"[EEG] Skybox transition complete. Current emotion: {currentEmotion}");
            }
        }
    }

    private void OnError(object sender, ErrorEventArgs e)
    {
        Debug.LogError($"[EEG] ❌ WebSocket error: {e.Message}");
    }

    private void OnClose(object sender, CloseEventArgs e)
    {
        Debug.Log($"[EEG] Disconnected from server. Code: {e.Code}, Reason: {e.Reason}");
        
        if (isDestroying || !isActiveAndEnabled)
        {
            Debug.Log("[EEG] Skip reconnect: GameObject is being destroyed or inactive");
            return;
        }
        
        StartReconnectCoroutine();
    }

    private void StartReconnectCoroutine()
    {
        if (!isActiveAndEnabled)
        {
            Debug.LogWarning("[EEG] Cannot start reconnect: GameObject is inactive");
            return;
        }
        
        if (reconnectCoroutine == null)
        {
            reconnectCoroutine = StartCoroutine(Reconnect());
        }
    }

    private void StopReconnectCoroutine()
    {
        if (reconnectCoroutine != null)
        {
            StopCoroutine(reconnectCoroutine);
            reconnectCoroutine = null;
        }
    }

    private IEnumerator Reconnect()
    {
        int retryCount = 0;
        while (!isDestroying && isActiveAndEnabled && (ws == null || !ws.IsAlive))
        {
            retryCount++;
            Debug.Log($"[EEG] Attempting to reconnect... ({retryCount})");
            yield return new WaitForSeconds(reconnectDelay);
            
            if (!isDestroying && isActiveAndEnabled)
            {
                Connect();
            }
        }
        
        reconnectCoroutine = null;
    }

    private void OnDisable()
    {
        StopAllCoroutines();
        reconnectCoroutine = null;
        crossfadeCoroutine = null;
    }

    private void OnDestroy()
    {
        isDestroying = true;
        
        StopAllCoroutines();
        reconnectCoroutine = null;
        crossfadeCoroutine = null;
        
        if (ws != null)
        {
            ws.OnOpen -= OnOpen;
            ws.OnMessage -= OnMessage;
            ws.OnError -= OnError;
            ws.OnClose -= OnClose;
            
            try
            {
                ws.Close();
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[EEG] Error closing WebSocket: {ex.Message}");
            }
            finally
            {
                ws = null;
            }
        }
        
        Debug.Log("[EEG] EmotionReceiver destroyed");
    }

    private void OnGUI()
    {
        if (!showDebugInfo || !isActiveAndEnabled) return;
        
        GUILayout.BeginArea(new Rect(10, 10, 350, 250));
        GUILayout.BeginVertical("box");
        
        GUILayout.Label("🧠 EEG Emotion Receiver", GUI.skin.box);
        GUILayout.Space(10);
        
        bool isConnected = ws != null && ws.IsAlive;
        GUILayout.Label($"🔗 Connection: {(isConnected ? "✅ Connected" : "❌ Disconnected")}");
        GUILayout.Label($"😊 Current Emotion: {currentEmotion.ToUpper()}");
        GUILayout.Label($"📊 Confidence: {currentConfidence:P1}");
        GUILayout.Label($"⏱️ Transition: {transitionProgress:P1}");
        
        if (isConnected)
        {
            GUI.color = Color.green;
            GUILayout.Label("● LIVE", GUI.skin.box);
            GUI.color = Color.white;
        }
        else
        {
            GUI.color = Color.red;
            GUILayout.Label("○ OFFLINE", GUI.skin.box);
            GUI.color = Color.white;
        }
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }

    [System.Serializable]
    private class EmotionData
    {
        public string emotion;
        public float confidence;
        public float transition_progress;
        public Probabilities probabilities;
        public float timestamp;
    }

    [System.Serializable]
    private class Probabilities
    {
        public float happy;
        public float sad;
        public float normal;
    }
}
