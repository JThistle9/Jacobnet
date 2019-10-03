import aiohttp
import asyncio
import uvicorn
import PIL.Image
import mtcnn
from mtcnn.mtcnn import MTCNN
import os
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1aCAzP0bIsHWTp1EQg1nWrWCOJpZdahIq'
export_file_name = 'bounding_box_model1.pkl'
classes = ['jacob', 'not']
not_saved = 0
jacob_saved = 0
path = Path(__file__).parent

app = Starlette(debug=True)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})

@app.route('/predict', methods=['POST'])
async def predict(request):
    img_bytes = await request.body()
    detector = MTCNN()
    result = detector.detect_faces(img_bytes)
    
    if (result != []):
        x, y, width, height = result[0]['box']
        x2, y2 = x+width, y+height
        image = Image.open(BytesIO(img_bytes))
        cropped_image = image.crop((x, y, x2, y2))
        cropped_image.save("./tmp/cropped_image.png")
        img = open_image("./tmp/cropped_image.png") 
        prediction = learn.predict(img, thresh=0.7)[0]
        if os.path.exists("./tmp/cropped_image.png"):
            os.remove("./tmp/cropped_image.png")
            print("cropped_image.png safely removed after use :)")
        else:
            print("The cropped_image.png did not exist :(")
    else:
        prediction = "not"
        
    return PlainTextResponse(str(prediction))

@app.route('/save/test/jacob', methods=['POST'])
async def savetestjacob(request):
    print('saving to ./data/test/jacob/')
    img_bytes = await request.body()
    image = PIL.Image.open(io.BytesIO(img_bytes))
    path = './data/test/jacob/'
    global jacob_saved
    jacob_saved = jacob_saved + 1
    filename = 'jacob_from_app_bb'+str(jacob_saved)+'.png'
    image.save(path + filename)
    return PlainTextResponse('saved file ' + path + filename)

@app.route('/save/test/not', methods=['POST'])
async def savetestnot(request):
    print('saving to ./data/test/not/')
    img_bytes = await request.body()
    image = PIL.Image.open(io.BytesIO(img_bytes))
    path = './data/test/not/'
    global not_saved
    not_saved = not_saved + 1
    filename = 'not_from_app_bb'+str(not_saved)+'.png'
    image.save(path+filename)
    return PlainTextResponse('saved file ' + path + filename)

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
