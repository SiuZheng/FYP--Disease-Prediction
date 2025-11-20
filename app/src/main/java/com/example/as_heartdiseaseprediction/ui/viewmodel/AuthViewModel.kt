package com.example.as_heartdiseaseprediction.ui.viewmodel

import com.google.firebase.auth.FirebaseAuth
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import com.google.firebase.auth.FirebaseAuthException

class AuthViewModel : ViewModel() {
    private val _isAuthenticated = MutableStateFlow(false)
    val isAuthenticated: StateFlow<Boolean> = _isAuthenticated.asStateFlow()
    
    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val auth: FirebaseAuth = FirebaseAuth.getInstance()

    init {
        // Check if user is already signed in
        _isAuthenticated.value = auth.currentUser != null
        
        // Add auth state listener
        auth.addAuthStateListener { firebaseAuth ->
            _isAuthenticated.value = firebaseAuth.currentUser != null
        }
    }

    fun signIn(email: String, password: String) {
        _isLoading.value = true
        _errorMessage.value = null
        
        auth.signInWithEmailAndPassword(email, password)
            .addOnCompleteListener { task ->
                _isLoading.value = false
                if (task.isSuccessful) {
                    _isAuthenticated.value = true
                    _errorMessage.value = null
                } else {
                    _isAuthenticated.value = false
                    _errorMessage.value = task.exception?.message ?: "Authentication failed"
                }
            }
    }

    fun signUp(fullName: String, email: String, password: String) {
        _isLoading.value = true
        _errorMessage.value = null
        
        auth.createUserWithEmailAndPassword(email, password)
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    // Update user profile with display name
                    val user = auth.currentUser
                    val profileUpdates = com.google.firebase.auth.UserProfileChangeRequest.Builder()
                        .setDisplayName(fullName)
                        .build()
                    
                    user?.updateProfile(profileUpdates)
                        ?.addOnCompleteListener { profileTask ->
                            _isLoading.value = false
                            if (profileTask.isSuccessful) {
                                _isAuthenticated.value = true
                                _errorMessage.value = null
                            } else {
                                _isAuthenticated.value = false
                                _errorMessage.value = profileTask.exception?.message ?: "Profile update failed"
                            }
                        }
                } else {
                    _isLoading.value = false
                    _isAuthenticated.value = false
                    _errorMessage.value = task.exception?.message ?: "Registration failed"
                }
            }
    }

    fun signOut() {
        auth.signOut()
        _isAuthenticated.value = false
        _errorMessage.value = null
    }

    fun clearError() {
        _errorMessage.value = null
    }

    override fun onCleared() {
        super.onCleared()
        // Remove auth state listener when ViewModel is cleared
        auth.removeAuthStateListener { }
    }
} 