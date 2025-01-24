package com.example.applayout.Marketplace

import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebView.setWebContentsDebuggingEnabled
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun WebViewScreen(url: String) {
    AndroidView(
        modifier = Modifier.fillMaxWidth(),
        factory = { context ->
            WebView(context).apply {
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = true
                settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                clearCache(true)
                clearHistory()
                setWebContentsDebuggingEnabled(true)
                webViewClient = WebViewClient()
                loadUrl(url)
            }
        }
    )
}
