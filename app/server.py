import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1-R8Zag0j-44MGtSM978alJSr8tIUSSmQ'
export_file_name = 'export.pkl'

classes = ['Accipiter ventralis',
  'Acropternis orthonyx',
  'Actitis macularius',
  'Aglaeactis cupripennis',
  'Aglaiocercus kingi',
  'Amazona mercenarius',
  'Ampelion rubrocristatus',
  'Anairetes agilis',
  'Anas andium',
  'Anas georgica',
  'Andigena nigrirostris',
  'Anhinga anihnga',
  'Anisognathus igniventris',
  'Anthus bogotensis',
  'Arremon assimilis',
  'Arremon brunneinucha',
  'Asio stygius',
  'Asthenes flammulata',
  'Asthenes fuliginosa',
  'Astragalinus psaltria',
  'Atlapetes pallidinucha',
  'Atlapetes schistaceus',
  'Aulacorhynchus prasinus',
  'Boissonneaua flavescens',
  'Buteo albigula',
  'Buteo platypterus',
  'Buthraupis montana',
  'Cacicus chrysonotus',
  'Campephilus pollens',
  'Catamblyrhynchus diadema',
  'Catamenia homochroa',
  'Catamenia inornata',
  'Catharus fuscater',
  'Catharus ustulatus',
  'Chaetocercus heliodor',
  'Chaetocercus mulsant',
  'Chalcostigma heteropogon',
  'Chamaepetes goudotii',
  'Chlorophonia pyrrhophrys',
  'Chlorornis riefferii',
  'Chlorospingus flavopectus',
  'Ciccaba albitarsis',
  'Cinclus leucocephalus',
  'Cinnycerthia olivascens',
  'Cinnycerthia unirufa',
  'Cistothorus platensis',
  'Cnemarchus erythropygius',
  'Cnemathraupis eximia',
  'Cnemoscopus rubrirostris',
  'Coeligena bonapartei',
  'Coeligena helianthea',
  'Coeligena torquata',
  'Colaptes rivolii',
  'Colaptes rubiginosus',
  'Colibri coruscans',
  'Colibri thalassinus',
  'Conirostrum albifrons',
  'Conirostrum rufum',
  'Conirostrum sitticolor',
  'Contopus cooperi',
  'Contopus fumigatus',
  'Coragyps atratus',
  'Coryphaspiza melanotis',
  'Creurgops verticalis',
  'Crotophaga ani',
  'Cyanocorax yncas',
  'Cyanolyca armillata',
  'Dendrocincla tyrannina',
  'Dendrocolaptes picumnus',
  'Diglossa albilatera',
  'Diglossa caerulescens',
  'Diglossa cyanea',
  'Diglossa humeralis',
  'Diglossa lafresnayii',
  'Doryfera ludovicae',
  'Drymophila caudata',
  'Dubusia taeniata',
  'Elaenia frantzii',
  'Empidonax alnorum',
  'Ensifera ensifera',
  'Eriocnemis cupreoventris',
  'Eriocnemis vestita',
  'Falco columbarius',
  'Falco sparverius',
  'Fulica americana',
  'Gallinago delicata',
  'Gallinago nobilis',
  'Gallinula galeata',
  'Geothlypis philadelphia',
  'Geranoaetus melanoleucus',
  'Glaucidium jardinii',
  'Grallaria quitensis',
  'Grallaria ruficapilla',
  'Grallaria rufula',
  'Grallaria squamigera',
  'Grallaricula nana',
  'Hapalopsittaca amazonina',
  'Haplophaedia aureliae',
  'Heliangelus amethysticollis',
  'Heliangelus mavors',
  'Hellmayrea gularis',
  'Hemitriccus granadensis',
  'Henicorhina leucophrys',
  'Hirundo rustica',
  'Icterus chrysater',
  'Iridosornis rufivertex',
  'Kleinothraupis atropileus',
  'Lafresnaya lafresnayi',
  'Leptasthenura andicola',
  'Leptopogon rufipectus',
  'Leptotila verreauxi',
  'Lesbia nuna',
  'Lesbia victoriae',
  'Lipaugus fuscocinereus',
  'Margarornis squamiger',
  'Mecocerculus leucophrys',
  'Mecocerculus poecilocercus',
  'Mecocerculus stictopterus',
  'Megascops albogularis',
  'Megascops choliba',
  'Merganetta armata',
  'Metallura tyrianthina',
  'Mionectes striaticollis',
  'Myadestes ralloides',
  'Myioborus ornatus',
  'Myiophobus flavicans',
  'Myiotheretes fumigatus',
  'Myiotheretes striaticollis',
  'Myiothlypis coronata',
  'Myiothlypis luteoviridis',
  'Myiothlypis nigrocristata',
  'Myoborus miniatus',
  'Nothocercus julius',
  'Nyctibius griseus',
  'Ochthoeca cinnamomeiventris',
  'Ochthoeca diadema',
  'Ochthoeca frontalis',
  'Ochthoeca fumicolor',
  'Ochthoeca rufipectoralis',
  'Opisthoprora euryptera',
  'Orochelidon murina',
  'Oxypogon guerinii',
  'Oxyura jamaicensis',
  'Pandion haliaetus',
  'Parabuteo leucorrhous',
  'Patagioenas fasciata',
  'Patagioenas subvinacea',
  'Patagona gigas',
  'Penelope montagnii',
  'Pharomachrus antisianus',
  'Pharomachrus auriceps',
  'Phimosus infuscatus',
  'Piaya cayana',
  'Picoides fumigatus',
  'Pionus tumultuosus',
  'Pipraeidea melanonota',
  'Pipreola arcuata',
  'Pipreola riefferii',
  'Piranga rubra',
  'Poecilotriccus ruficeps',
  'Pseudocolaptes boissonneautii',
  'Pseudospingus verticalis',
  'Pseudotriccus ruficeps',
  'Pterophanes cyanopterus',
  'Pyrrhomyias cinnamomeus',
  'Pyrrhura calliptera',
  'Rallus semiplumbeus',
  'Ramphomicron microrhynchum',
  'Riparia riparia',
  'Sayornis nigricans',
  'Scytalopus griseicollis',
  'Scytalopus latrans',
  'Sericossypha albocristata',
  'Setophaga fusca',
  'Setophaga ruticilla',
  'Spizaetus isidori',
  'Steatornis caripensis',
  'Streptoprocne zonaris',
  'Synallaxis azarae',
  'Synallaxis subpudica',
  'Synallaxis unirufa',
  'Systelura longirostris',
  'Tachyphonus rufus',
  'Tangara heinei',
  'Tangara nigroviridis',
  'Tangara vassorii',
  'Thlypopsis superciliaris',
  'Thraupis cyanocephala',
  'Thripadectes flammulatus',
  'Tigrisoma lineatum',
  'Troglodytes aedon',
  'Troglodytes solstitialis',
  'Trogon personatus',
  'Turdus fuscater',
  'Turdus serranus',
  'Tyto alba',
  'Vultur gryphus',
  'Xiphocolaptes promeropirhynchus',
  'Xiphorhynchus triangularis',
  'Zenaida auriculata',
  'Zonotrichia capensis']
path = Path(__file__).parent

app = Starlette()
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


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
