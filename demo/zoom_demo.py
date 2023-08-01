import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 확대하려는 영역의 좌표와 크기
zoom_x = 100
zoom_y = 100
zoom_width = 100
zoom_height = 150

im = Image.open('demo/demo.png')

# Create figures and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 두 개의 subplot 생성

# Display the original image in the first subplot (ax1)
ax1.imshow(im, origin='upper')
ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

# Create a Rectangle patch in the first subplot (ax1)
rect = patches.Rectangle((zoom_x, zoom_y), zoom_width, zoom_height, linewidth=1, edgecolor='r', facecolor='none')
ax1.add_patch(rect)

# ax1의 테두리 비활성화
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Display the enlarged image in the second subplot (ax2)
ax2.imshow(im, origin='upper')
ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)


# Create a Rectangle patch and specify the zoomed-in area in the second subplot (ax2)
zoomed_rect = patches.Rectangle((zoom_x, zoom_y), zoom_width, zoom_height, linewidth=1, edgecolor='r', facecolor='none')
ax2.add_patch(zoomed_rect)

# 확대된 영역 설정
ax2.set_xlim(zoom_x, zoom_x + zoom_width)
ax2.set_ylim(zoom_y + zoom_height, zoom_y) # 세로 방향을 반전시키기 위해 높이를 더해줍니다.

# 두 번째 그림(ax2)의 테두리 스타일 설정
for spine in ax2.spines.values():
    spine.set_linewidth(1)  # 테두리 두께 설정
    spine.set_edgecolor('r')  # 테두리 색상 설정

plt.savefig('demo/result.png')
