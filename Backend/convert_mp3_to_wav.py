import os
import subprocess

def convert_mp3_to_wav(input_dir='.'):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.mp3'):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.wav'
            output_path = os.path.join(input_dir, output_filename)

            # ffmpeg command
            command = [
                'ffmpeg',
                '-i', input_path,
                '-ar', '16000',    # 设置采样率为 16000 Hz
                '-ac', '1',        # 设置为单声道
                output_path
            ]

            try:
                print(f'正在转换: {filename} → {output_filename}')
                subprocess.run(command, check=True)
                print('✅ 转换完成\n')
            except subprocess.CalledProcessError:
                print(f'❌ 转换失败: {filename}\n')

if __name__ == '__main__':
    convert_mp3_to_wav()