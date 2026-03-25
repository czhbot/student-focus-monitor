<script>
export default {
  onLaunch: function() {
    console.log('App Launch')
  },
  onShow: function() {
    console.log('App Show')
  },
  onHide: function() {
    console.log('App Hide')
  }
}
</script>

<style>
page {
  background-color: #f8fafc;
  background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
  background-size: 40rpx 40rpx;
  min-height: 100vh;
}

.container {
  padding: 24rpx;
}

.glass-panel {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(24rpx);
  border: 1px solid rgba(255, 255, 255, 0.5);
  box-shadow: 0 16rpx 64rpx rgba(37, 114, 188, 0.08);
}

.text-brand {
  color: #2572BC;
}

.bg-brand {
  background-color: #2572BC;
}

.uni-page-head {
  background: linear-gradient(135deg, #4a9eda 0%, #2572BC 50%, #1a5a94 100%) !important;
  box-shadow: 0 4rpx 20rpx rgba(37, 114, 188, 0.4);
}

.uni-page-head__title {
  font-weight: 600 !important;
  background: linear-gradient(135deg, #ffffff 0%, #c4e4f7 50%, #ffffff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.liquid-glass-btn {
  position: relative;
  background: linear-gradient(135deg, 
    rgba(255, 255, 255, 0.4) 0%, 
    rgba(255, 255, 255, 0.1) 50%,
    rgba(255, 255, 255, 0.3) 100%);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.5);
  box-shadow: 
    0 8rpx 32rpx rgba(37, 114, 188, 0.15),
    inset 0 1px 0 rgba(255, 255, 255, 0.5),
    inset 0 -1px 0 rgba(0, 0, 0, 0.05);
  overflow: hidden;
  transition: all 0.3s ease;
}

.liquid-glass-btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, 
    transparent, 
    rgba(255, 255, 255, 0.4), 
    transparent);
  transition: left 0.5s ease;
}

.liquid-glass-btn:active::before {
  left: 100%;
}

.liquid-glass-btn:active {
  transform: scale(0.98);
  box-shadow: 
    0 4rpx 16rpx rgba(37, 114, 188, 0.2),
    inset 0 1px 0 rgba(255, 255, 255, 0.5);
}

.liquid-glass-btn-primary {
  background: linear-gradient(135deg, 
    rgba(74, 158, 218, 0.7) 0%, 
    rgba(37, 114, 188, 0.5) 50%,
    rgba(26, 90, 148, 0.6) 100%);
  border: 1px solid rgba(37, 114, 188, 0.4);
}

.liquid-glass-btn-primary .btn-text {
  color: white;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
}

.liquid-glass-btn-danger {
  background: linear-gradient(135deg, 
    rgba(244, 63, 94, 0.5) 0%, 
    rgba(225, 29, 72, 0.3) 50%,
    rgba(244, 63, 94, 0.4) 100%);
  border: 1px solid rgba(244, 63, 94, 0.3);
}

.liquid-glass-btn-danger .btn-text {
  color: #be123c;
  text-shadow: 0 1px 2px rgba(255, 255, 255, 0.3);
}

.liquid-glass-btn-secondary {
  background: linear-gradient(135deg, 
    rgba(148, 163, 184, 0.3) 0%, 
    rgba(100, 116, 139, 0.2) 50%,
    rgba(148, 163, 184, 0.25) 100%);
  border: 1px solid rgba(148, 163, 184, 0.3);
}

.liquid-glass-btn-secondary .btn-text {
  color: #334155;
}
</style>
