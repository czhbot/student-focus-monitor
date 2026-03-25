<template>
  <view class="config-container">
    <view class="config-card glass-panel">
      <view class="config-header">
        <text class="config-title">服务器配置</text>
        <text class="config-desc">请输入运行监控系统的服务器IP地址</text>
      </view>
      
      <view class="config-form">
        <view class="form-item">
          <text class="form-label">服务器IP地址</text>
          <input 
            class="form-input" 
            type="text" 
            v-model="serverIp" 
            placeholder="例如: 192.168.1.100"
          />
        </view>
        
        <view class="form-item">
          <text class="form-label">端口号</text>
          <input 
            class="form-input" 
            type="number" 
            v-model="serverPort" 
            placeholder="默认: 5000"
          />
        </view>
        
        <view class="form-tips">
          <text class="tips-icon">💡</text>
          <text class="tips-text">请确保手机和服务器在同一局域网内</text>
        </view>
        
        <view class="form-actions">
          <button class="btn-save liquid-glass-btn liquid-glass-btn-primary" @click="saveConfig">
            <text class="btn-text">保存配置</text>
          </button>
          <button class="btn-test liquid-glass-btn liquid-glass-btn-secondary" @click="testConnection">
            <text class="btn-text">测试连接</text>
          </button>
        </view>
      </view>
    </view>
    
    <view class="quick-tips glass-panel">
      <text class="tips-title">快速指南</text>
      <view class="tips-list">
        <view class="tips-item">
          <text class="tips-num">1</text>
          <text class="tips-content">在电脑上运行监控系统，查看控制台显示的IP地址</text>
        </view>
        <view class="tips-item">
          <text class="tips-num">2</text>
          <text class="tips-content">确保手机连接到与电脑相同的WiFi网络</text>
        </view>
        <view class="tips-item">
          <text class="tips-num">3</text>
          <text class="tips-content">在上方输入IP地址并保存，然后返回主页开始监控</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const serverIp = ref('')
const serverPort = ref('5000')

const loadConfig = () => {
  const savedIp = uni.getStorageSync('server_ip')
  const savedPort = uni.getStorageSync('server_port')
  
  if (savedIp) serverIp.value = savedIp
  if (savedPort) serverPort.value = savedPort
}

const saveConfig = () => {
  if (!serverIp.value) {
    uni.showToast({ title: '请输入服务器IP地址', icon: 'none' })
    return
  }
  
  uni.setStorageSync('server_ip', serverIp.value)
  uni.setStorageSync('server_port', serverPort.value)
  
  // 核心修复：彻底删除了 uni.$emit，解决死循环刷新
  uni.showToast({ title: '配置已保存', icon: 'success' })
  
  setTimeout(() => {
    // 增加路由栈判断，防止在H5中刷新设置页后直接点击保存导致无法返回
    const pages = getCurrentPages()
    if (pages.length > 1) {
      uni.navigateBack()
    } else {
      // 请确认下方的路径是你首页的真实路径
      uni.redirectTo({ url: '/pages/index/index' }) 
    }
  }, 1000)
}

const testConnection = async () => {
  if (!serverIp.value) {
    uni.showToast({ title: '请输入服务器IP地址', icon: 'none' })
    return
  }
  
  uni.showLoading({ title: '测试连接中...' })
  
  try {
    const response = await new Promise((resolve, reject) => {
      uni.request({
        url: `http://${serverIp.value}:${serverPort.value || 5000}/`,
        method: 'GET',
        timeout: 5000,
        success: (res) => resolve(res),
        fail: (err) => reject(err)
      })
    })
    
    uni.hideLoading()
    
    if (response.statusCode === 200) {
      uni.showModal({ title: '连接成功', content: '服务器连接正常，可以开始监控', showCancel: false })
    } else {
      uni.showToast({ title: '服务器响应异常', icon: 'none' })
    }
  } catch (error) {
    uni.hideLoading()
    uni.showModal({ title: '连接失败', content: '无法连接到服务器，请检查IP地址和网络连接', showCancel: false })
  }
}

onMounted(() => {
  loadConfig()
})
</script>

<style scoped>
/* 原有的样式保持不变... */
.config-container { min-height: 100vh; padding: 24rpx; }
.config-card { padding: 32rpx; border-radius: 24rpx; margin-bottom: 24rpx; background: rgba(255, 255, 255, 0.95); box-shadow: 0 8rpx 32rpx rgba(14, 165, 233, 0.08); }
.config-header { margin-bottom: 32rpx; text-align: center; }
.config-title { font-size: 36rpx; font-weight: 700; color: #1e293b; display: block; margin-bottom: 8rpx; }
.config-desc { font-size: 26rpx; color: #64748b; }
.config-form { display: flex; flex-direction: column; gap: 24rpx; }
.form-item { display: flex; flex-direction: column; gap: 12rpx; }
.form-label { font-size: 28rpx; color: #334155; font-weight: 500; }
.form-input { height: 88rpx; background: #f8fafc; border: 2rpx solid #e2e8f0; border-radius: 16rpx; padding: 0 24rpx; font-size: 30rpx; color: #1e293b; }
.form-input:focus { border-color: #0ea5e9; background: white; }
.form-tips { display: flex; align-items: center; gap: 12rpx; padding: 20rpx; background: #fefce8; border-radius: 12rpx; border: 1px solid #fef08a; }
.tips-icon { font-size: 32rpx; }
.tips-text { font-size: 24rpx; color: #854d0e; }
.form-actions { display: flex; gap: 20rpx; margin-top: 16rpx; }
.btn-save, .btn-test { flex: 1; height: 88rpx; border-radius: 16rpx; display: flex; align-items: center; justify-content: center; }
.btn-text { font-size: 30rpx; font-weight: 500; }
.quick-tips { padding: 32rpx; border-radius: 24rpx; background: rgba(255, 255, 255, 0.95); box-shadow: 0 8rpx 32rpx rgba(14, 165, 233, 0.08); }
.tips-title { font-size: 30rpx; font-weight: 600; color: #334155; display: block; margin-bottom: 24rpx; }
.tips-list { display: flex; flex-direction: column; gap: 20rpx; }
.tips-item { display: flex; align-items: flex-start; gap: 16rpx; }
.tips-num { width: 40rpx; height: 40rpx; background: #0ea5e9; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24rpx; font-weight: 600; flex-shrink: 0; }
.tips-content { font-size: 26rpx; color: #64748b; line-height: 1.6; }
</style>