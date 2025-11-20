package com.example.as_heartdiseaseprediction.data.preferences

import android.content.Context
import android.content.SharedPreferences

class SosPreferences(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    
    companion object {
        private const val PREFS_NAME = "sos_preferences"
        private const val KEY_EMERGENCY_NUMBER = "emergency_number"
        private const val KEY_AUTO_SEND_ENABLED = "auto_send_enabled"
        private const val KEY_TIMER_DELAY = "timer_delay"
        
        // Default values
        private const val DEFAULT_EMERGENCY_NUMBER = "0123456789"
        private const val DEFAULT_AUTO_SEND_ENABLED = false
        private const val DEFAULT_TIMER_DELAY = 5 // seconds
    }
    
    var emergencyNumber: String
        get() = prefs.getString(KEY_EMERGENCY_NUMBER, DEFAULT_EMERGENCY_NUMBER) ?: DEFAULT_EMERGENCY_NUMBER
        set(value) = prefs.edit().putString(KEY_EMERGENCY_NUMBER, value).apply()
    
    var isAutoSendEnabled: Boolean
        get() = prefs.getBoolean(KEY_AUTO_SEND_ENABLED, DEFAULT_AUTO_SEND_ENABLED)
        set(value) = prefs.edit().putBoolean(KEY_AUTO_SEND_ENABLED, value).apply()
    
    var timerDelay: Int
        get() = prefs.getInt(KEY_TIMER_DELAY, DEFAULT_TIMER_DELAY)
        set(value) = prefs.edit().putInt(KEY_TIMER_DELAY, value).apply()
    
    fun clearAll() {
        prefs.edit().clear().apply()
    }
}
