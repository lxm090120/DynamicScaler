from dataclasses import dataclass
from typing import Dict

i2v_sp_input_dict = {
    "ocean_world": {
        "pano_image_path": "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_ocean_world.png",
        "prompt": "Immersive underwater view of vibrant coral reef, crystal clear water filled with marine life, powerful shafts of sunlight, sea turtles and dolphins swimming through blue waters, colorful tropical fish darting between coral formations, rich coral gardens stretching into deep blue distance, water particles and small bubbles catching sunlight",
        "phi_prompt_dict": {
            90: "Underwater view of ocean, fish swimming in ocean, sun rays piercing through moving water",
            75: "Underwater view of ocean, fish swimming in ocean, sun rays piercing through moving water",
            60: "Underwater view of ocean, crystal clear water filled with marine life, powerful shafts of sunlight, sea turtles and dolphins and colorful fish swimming through blue waters",
            45: "Underwater view of ocean, crystal clear water filled with marine life, powerful shafts of sunlight, sea turtles and dolphins and colorful fish swimming through blue waters",

            0: "Immersive underwater view of vibrant coral reef, crystal clear water filled with marine life, powerful shafts of sunlight, sea turtles and dolphins swimming through blue waters, colorful tropical fish darting between coral formations, rich coral gardens stretching into deep blue distance, water particles and small bubbles catching sunlight",

            -45: "Coral reef ecosystem viewed underwater, branching corals in vivid colors swaying with water movement, schools of butterfly and angelfish weaving through coral formations, sea anemones dancing in current, floating particles in water catching light",
            -60: "seafloor with coral view underwater ",
            -75: "seafloor with coral view underwater ",
            -90: "seafloor with coral view underwater ",
        }
    },

    "4k4dgen_greenland": {
        "pano_image_path": "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_greenland_512-256.png",

        "prompt": "A sunny landscape with green hills and dirt paths, fluffy white clouds moving in bright blue sky",

        "phi_prompt_dict": {
            90: "fluffy white clouds moving in bright blue sky",
            75: "fluffy white clouds moving in bright blue sky",
            60: "fluffy white clouds moving in bright blue sky",
            45: "fluffy white clouds moving in bright blue sky above hills",

            0: "A sunny landscape with green hills and dirt paths, fluffy white clouds moving in bright blue sky ",

            -45: "A sunny landscape with green hills and dirt paths",
            -60: "A sunny landscape with green hills and dirt paths",
            -75: "A sunny landscape with green hills and dirt paths",
            -90: "A sunny landscape with green hills and dirt paths",
        }
    },

    "real_greenland": {
        "prompt": "clouds moving in bright blue sky, green hills with winding country roads, small rural village with stone houses nestled in valley, scattered trees and hedgerows, mixed grassland and farmland",
        "pano_image_path": "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano360_real_blue_sky_green_grass.jpg",

        "phi_prompt_dict": {
            90: "blue sky with scattered cumulus clouds, intense sunlight breaking through clouds creating light rays",
            75: "blue sky with scattered cumulus clouds, intense sunlight breaking through clouds creating light rays",
            60: "blue sky with scattered cumulus clouds, intense sunlight breaking through clouds creating light rays",
            45: "blue sky with scattered cumulus clouds, intense sunlight breaking through clouds creating light rays",

            0: "green hills with winding country roads, small rural village with stone houses nestled in valley, scattered trees and hedgerows, mixed grassland and farmland, blue sky with dramatic clouds above in bright summer day",

            -45: "Verdant grass meadows and agricultural fields from elevated view, network of curved country roads connecting small settlements, patches of woodland, varied terrain with gentle slopes",
            -60: "Verdant grass meadows and agricultural fields from elevated view, network of curved country roads connecting small settlements, patches of woodland, varied terrain with gentle slopes",
            -75: "Verdant grass meadows and agricultural fields from elevated view, network of curved country roads connecting small settlements, patches of woodland, varied terrain with gentle slopes",
            -90: "Verdant grass meadows and agricultural fields from elevated view, network of curved country roads connecting small settlements, patches of woodland, varied terrain with gentle slopes",
        }
    },

    "4k4dgen_volcano": {
        "prompt": "volcanoes erupting at sunset, flowing lava rivers cutting through dark volcanic landscape, glowing magma streams between black volcanic hills, ash clouds billowing in distance",
        "pano_image_path": "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_volcano.png",
        "phi_prompt_dict": {
            90: "dusk sky with billowing ash clouds in deep purples and oranges at sunset",
            75: "dusk sky with billowing ash clouds in deep purples and oranges at sunset",
            60: "volcanoes erupting against at sunset, ash clouds billowing towards dusk sky",
            45: "volcanoes erupting against at sunset, ash clouds billowing towards dusk sky",

            0: "volcanoes erupting against at sunset, flowing lava rivers cutting through dark volcanic landscape, glowing magma streams between black volcanic hills, ash clouds billowing towards dusk sky in distance",

            -45: "lava flowing through volcanic terrain from elevated view, black volcanic rock formations with glowing cracks, cooling lava creating new land formations",
            -60: "lava flowing through volcanic terrain from elevated view, black volcanic rock formations with glowing cracks, cooling lava creating new land formations",
            -75: "lava flowing through volcanic terrain from elevated view, black volcanic rock formations with glowing cracks, cooling lava creating new land formations",
            -90: "Red mountain with black volcanic rocks",
        }
    },

    "4k4dgen_firework": {
        "prompt": "A vibrant cityscape at sunset with fireworks bursting in the sky, while cars traffics move along the illuminated roads below",
        "pano_image_path": "/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_fireworkcity_2048-1024.png",
        "phi_prompt_dict": {
            90: "A vibrant cityscape at sunset with fireworks bursting in the sky",
            75: "A vibrant cityscape at sunset with fireworks bursting in the sky",
            60: "A vibrant cityscape at sunset with fireworks bursting in the sky",
            45: "A vibrant cityscape at sunset with fireworks bursting in the sky",

            0: "A vibrant cityscape at sunset with fireworks bursting in the sky, while cars traffics move along the illuminated roads below",

            -45: "cars traffics move steadily along the illuminated roads in a vibrant cityscape",
            -60: "cars traffics move steadily along the illuminated roads in a vibrant cityscape",
            -75: "cars traffics move steadily along the illuminated roads in a vibrant cityscape",
            -90: "cars traffics move steadily along the illuminated roads in a vibrant cityscape",
        }
    }
}

@dataclass
class I2V_SP_INPUT:
    prompt: str
    pano_image_path: str
    phi_prompt_dict: Dict[int, str]
    window_multi_prompt_dict: Dict[float, str] = None

II_ocean_world = I2V_SP_INPUT(
    pano_image_path="/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_ocean_world.png",
    prompt = "Immersive underwater view of vibrant coral reef, crystal clear water filled with marine life, powerful shafts of sunlight, sea turtles and dolphins swimming through blue waters, colorful tropical fish darting between coral formations, rich coral gardens stretching into deep blue distance, water particles and small bubbles catching sunlight",
    phi_prompt_dict = {
        90: "Underwater view of ocean, fish swimming in ocean, sun rays piercing through moving water",
        75: "Underwater view of ocean, fish swimming in ocean, sun rays piercing through moving water",
        60: "Underwater view of ocean, crystal clear water filled with marine life, powerful shafts of sunlight, sea turtles and dolphins and colorful fish swimming through blue waters",
        45: "Underwater view of ocean, crystal clear water filled with marine life, powerful shafts of sunlight, sea turtles and dolphins and colorful fish swimming through blue waters",

        0: "Immersive underwater view of vibrant coral reef, crystal clear water filled with marine life, powerful shafts of sunlight, sea turtles and dolphins swimming through blue waters, colorful tropical fish darting between coral formations, rich coral gardens stretching into deep blue distance, water particles and small bubbles catching sunlight",

        -45: "Coral reef ecosystem viewed underwater, branching corals in vivid colors swaying with water movement, schools of butterfly and angelfish weaving through coral formations, sea anemones dancing in current, floating particles in water catching light",
        -60: "seafloor with coral view underwater ",
        -75: "seafloor with coral view underwater ",
        -90: "seafloor with coral view underwater ",
    }
)

II_greenland_4k4dgen = I2V_SP_INPUT(
    pano_image_path="/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_greenland_512-256.png",
    prompt="A sunny landscape with green hills and dirt paths, fluffy white clouds moving in bright blue sky",
    phi_prompt_dict={
        90: "fluffy white clouds moving in bright blue sky",
        75: "fluffy white clouds moving in bright blue sky",
        60: "fluffy white clouds moving in bright blue sky",
        45: "fluffy white clouds moving in bright blue sky above hills",

        0: "A sunny landscape with green hills and dirt paths, fluffy white clouds moving in bright blue sky ",

        -45: "A sunny landscape with green hills and dirt paths",
        -60: "A sunny landscape with green hills and dirt paths",
        -75: "A sunny landscape with green hills and dirt paths",
        -90: "A sunny landscape with green hills and dirt paths",
    }
)

II_greenland_real = I2V_SP_INPUT(
    prompt="clouds moving in bright blue sky, green hills with winding country roads, small rural village with stone houses nestled in valley, scattered trees and hedgerows, mixed grassland and farmland",
    pano_image_path="/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano360_real_blue_sky_green_grass.jpg",
    phi_prompt_dict={
        90: "blue sky with scattered cumulus clouds, intense sunlight breaking through clouds creating light rays",
        75: "blue sky with scattered cumulus clouds, intense sunlight breaking through clouds creating light rays",
        60: "blue sky with scattered cumulus clouds, intense sunlight breaking through clouds creating light rays",
        45: "blue sky with scattered cumulus clouds, intense sunlight breaking through clouds creating light rays",

        0: "green hills with winding country roads, small rural village with stone houses nestled in valley, scattered trees and hedgerows, mixed grassland and farmland, blue sky with dramatic clouds above in bright summer day",

        -45: "Verdant grass meadows and agricultural fields from elevated view, network of curved country roads connecting small settlements, patches of woodland, varied terrain with gentle slopes",
        -60: "Verdant grass meadows and agricultural fields from elevated view, network of curved country roads connecting small settlements, patches of woodland, varied terrain with gentle slopes",
        -75: "Verdant grass meadows and agricultural fields from elevated view, network of curved country roads connecting small settlements, patches of woodland, varied terrain with gentle slopes",
        -90: "Verdant grass meadows and agricultural fields from elevated view, network of curved country roads connecting small settlements, patches of woodland, varied terrain with gentle slopes",
    }
)

II_volcano_4k4dgen = I2V_SP_INPUT(
    prompt="volcanoes erupting at sunset, smoke clouds rolling and billowing towards the sky, flowing lava rivers cutting through dark volcanic landscape, glowing magma streams between black volcanic hills, ash clouds billowing in distance",
    pano_image_path="/home/jxliu/test/FIFO-Diffusion_public/input_pano/pano_volcano.png",
    phi_prompt_dict={
        90: "dusk sky with billowing ash clouds in deep purples and oranges at sunset",
        75: "dusk sky with billowing ash clouds in deep purples and oranges at sunset",
        60: "volcanoes erupting against at sunset, ash clouds billowing towards dusk sky",
        45: "volcanoes erupting against at sunset, ash clouds billowing towards dusk sky",

        0: "volcanoes erupting against at sunset, flowing lava rivers cutting through dark volcanic landscape, glowing magma streams between black volcanic hills, ash clouds billowing towards dusk sky in distance",

        -45: "lava flowing through volcanic terrain from elevated view, black volcanic rock formations with glowing cracks, cooling lava creating new land formations",
        -60: "lava flowing through volcanic terrain from elevated view, black volcanic rock formations with glowing cracks, cooling lava creating new land formations",
        -75: "lava flowing through volcanic terrain from elevated view, black volcanic rock formations with glowing cracks, cooling lava creating new land formations",
        -90: "Red mountain with black volcanic rocks",
    }
)

II_fireworks_4k4dgen = I2V_SP_INPUT(
    prompt="A vibrant cityscape at sunset with fireworks bursting in the sky, while cars traffics move along the streets in city",
    pano_image_path="/home/jxliu/test/FIFO-Diffusion_public/input_pano/pano_fireworkcity.png",
    phi_prompt_dict={
        90: "A vibrant cityscape at sunset with fireworks bursting in the sky",
        75: "A vibrant cityscape at sunset with fireworks bursting in the sky",
        60: "A vibrant cityscape at sunset with fireworks bursting in the sky",
        45: "A vibrant cityscape at sunset with fireworks bursting in the sky",

        0: "A vibrant cityscape at sunset with fireworks bursting in the sky, while cars traffics move along the illuminated roads below",

        -45: "cars traffics move steadily along the illuminated roads in a vibrant cityscape",
        -60: "cars traffics move steadily along the illuminated roads in a vibrant cityscape",
        -75: "cars traffics move steadily along the illuminated roads in a vibrant cityscape",
        -90: "cars traffics move steadily along the illuminated roads in a vibrant cityscape",
    }
)

II_beach_4k4dgen = I2V_SP_INPUT(
    prompt="Dynamic waves rushing onto beach, blue ocean meeting warm sand, water with white foam spreading across shoreline, sunlight sparkling on water surface",
    pano_image_path="/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_beachwave.jpg",
    phi_prompt_dict={
        90: "clear blue sky",
        75: "clear blue sky",
        60: "clear blue sky",
        45: "clear blue sky",

        0: "Dynamic waves rushing onto beach, blue ocean meeting warm sand, water with white foam spreading across shoreline, sunlight sparkling on water surface",

        -45: "dynamic sea wave rushing onto the beach, water with foam spreading across wet sand, sunlight sparkling on water surface",
        -60: "wet smooth sand on beach",
        -75: "wet smooth sand",
        -90: "smooth sand",
    },
    # window_multi_prompt_dict = {
    #     0.0: "blue sky with white clouds moving",
    #     0.3: "azure sky meets turquoise ocean at perfect horizon line, white clouds moving in sky, crystal clear watee waves rolling onto the shore with golden sand beach",
    #     0.6: "Clear ocean water with gentle waves rolling onto golden beach",
    # }
)

II_wave = I2V_SP_INPUT(
    prompt="Massive green blue ocean wave, dynamic ocean spray, dynamic water motion, breaking waves with white foam, seabirds in motion, turquoise water transparency, distant ocean horizon",
    pano_image_path="/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_surfing_1.png",
    phi_prompt_dict = {
        90: "Clear light blue sky",
        75: "Clear light blue sky",
        60: "Clear light blue sky",
        45: "Massive green blue ocean wave, dynamic ocean spray, dynamic water motion, breaking waves with white foam, seabirds in motion, turquoise water transparency, distant ocean horizon",

        0:  "Massive green blue ocean wave, dynamic ocean spray, dynamic water motion, breaking waves with white foam, seabirds in motion, turquoise water transparency, distant ocean horizon",

        -45: "green blue ocean with waves and swirling foam patterns",
        -60: "green blue ocean with waves",
        -75: "green blue ocean water",
        -90: "green blue ocean water",
    }
)

II_fireworks_flux = I2V_SP_INPUT(
    prompt="A vibrant cityscape at sunset with fireworks bursting in the sky, while cars traffics move along the streets in city",
    pano_image_path="/home/ljx/_temp/fifo_free_traj_turbo_test/datasets/equirect_panorama/pano_fireworkcity_flux.png",
    phi_prompt_dict={
        90: "A vibrant cityscape at sunset with fireworks bursting in the sky",
        75: "A vibrant cityscape at sunset with fireworks bursting in the sky",
        60: "A vibrant cityscape at sunset with fireworks bursting in the sky",
        45: "A vibrant cityscape at sunset with fireworks bursting in the sky",

        0: "A vibrant cityscape at sunset with fireworks bursting in the sky",

        -45: "A vibrant cityscape, cars traffics move steadily along the street in a vibrant cityscape",
        -60: "A vibrant cityscape, cars traffics move steadily along the street in a vibrant cityscape",
        -75: "A vibrant cityscape, cars traffics move steadily along the street in a vibrant cityscape",
        -90: "A vibrant cityscape, cars traffics move steadily along the street in a vibrant cityscape",
    }
)



