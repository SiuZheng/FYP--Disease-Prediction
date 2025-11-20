package com.example.as_heartdiseaseprediction.data.preferences

import android.content.Context
import android.content.SharedPreferences

class OnboardingPreferences(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    
    companion object {
        private const val PREFS_NAME = "onboarding_preferences"
        // Key for tracking if user has seen and acknowledged the disclaimer
        private const val KEY_DISCLAIMER_ACKNOWLEDGED = "disclaimer_acknowledged"
        // Key for tracking if user has chosen "Don't show again"
        private const val KEY_DONT_SHOW_AGAIN = "dont_show_again"
        
        // Default values
        private const val DEFAULT_DISCLAIMER_ACKNOWLEDGED = false
        private const val DEFAULT_DONT_SHOW_AGAIN = false
    }
    
    /**
     * Whether the user has acknowledged the disclaimer
     */
    var isDisclaimerAcknowledged: Boolean
        get() = prefs.getBoolean(KEY_DISCLAIMER_ACKNOWLEDGED, DEFAULT_DISCLAIMER_ACKNOWLEDGED)
        set(value) = prefs.edit().putBoolean(KEY_DISCLAIMER_ACKNOWLEDGED, value).apply()
    
    /**
     * Whether the user has chosen "Don't show again"
     */
    var dontShowAgain: Boolean
        get() = prefs.getBoolean(KEY_DONT_SHOW_AGAIN, DEFAULT_DONT_SHOW_AGAIN)
        set(value) = prefs.edit().putBoolean(KEY_DONT_SHOW_AGAIN, value).apply()
    
    /**
     * Check if onboarding should be shown
     * Returns true if user hasn't chosen "Don't show again"
     * This means the onboarding will show every time unless explicitly disabled
     */
    fun shouldShowOnboarding(): Boolean {
        return !dontShowAgain
    }
    
    /**
     * Acknowledge disclaimer without disabling onboarding
     * This is called when user clicks "I Understand" without checking "Don't show again"
     */
    fun acknowledgeDisclaimer() {
        prefs.edit()
            .putBoolean(KEY_DISCLAIMER_ACKNOWLEDGED, true)
            .apply()
    }
    
    /**
     * Reset onboarding state (useful for testing or if user wants to see onboarding again)
     */
    fun resetOnboarding() {
        prefs.edit()
            .putBoolean(KEY_DISCLAIMER_ACKNOWLEDGED, false)
            .putBoolean(KEY_DONT_SHOW_AGAIN, false)
            .apply()
    }
    
    /**
     * Clear all onboarding preferences
     */
    fun clearAll() {
        prefs.edit().clear().apply()
    }
}
