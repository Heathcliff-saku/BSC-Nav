from openai import OpenAI
import json
from io import BytesIO
import base64
from PIL import Image
# from qwen_vl_utils import process_vision_info


def imagenary_helper_visaug(client, text_prompt, vis):
    base64_images = image_to_base64(vis)
    completion = client.chat.completions.create(
      model="gpt-4o",
      timeout=500,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
          {"type": "text", "text": """
          You are an expert in generating prompts for text-to-image models. 
          Your task is to enhance a given original goal description, which often only mentions a general category or a simple phrase describing the target object, by incorporating detailed context from four current scene images. You will create a more imaginative and contextually enriched description, using elements observed in the scene to create a coherent and vivid visual. This description will be used to guide a text-to-image model to generate an image that aligns with the style and context of the current scene.
          It is crucial that the target object remains the primary visual focus of the image. To complete this task effectively, it's suggested to follow these steps:
          1. **Understand the Environment**: Extract and comprehend details from the provided observation images, such as the overall style, decoration, and elements of the scene. For example, is this a modern home, a classical residence, an exhibition hall, or an office? Consider the style in detail, and understand the context of the environment.
          2. **Expand the Original Description **: Based on the scene analysis from step 1, enrich the original description with finer details. This may include materials, colors, textures, placement, and environmental elements surrounding the target object.
          3. **Maintain Visual Focus **: Ensure that any additional context or background details do not overshadow the main target object. The primary subject should remain the focal point of the generated image, using language that emphasizes its prominence in the scene.

          **Guidelines for Creating Enhanced Descriptions: **
          1. Details: Include sensory details like colors, textures, lighting, and reflections.
          2. Background Elements: Add appropriate background elements that complement the scene without detracting from the focus of the image.
          Focus Phrasing: Use language that naturally draws attention to the target object (e.g., "centered," "prominently placed," "as the focal point").
          3. Balance: Strike a balance between richness and simplicity. The target object or scene should always dominate the final image.
          **Enhanced Description Output Requirements: **
          1. Provide a refined and detailed description in English.
          2. Ensure the enhancement creates a vivid, coherent, and engaging scene, supporting the original description.
          3. Avoid overly complex narratives or elements that distract from the primary object or scene.
          4. Keep the description concise, limiting it to 70 words or less.
         
          **Examples:**
          1.
          Original Goal Description: A green vase. 
          Enhanced Description: A vibrant green ceramic vase, with a glossy, smooth surface, placed centrally on a polished wooden table. Soft natural light illuminates the vase from the large window behind it, casting gentle shadows on the table. The surrounding room is decorated in minimalist modern style with neutral tones, ensuring the vase is the central focal point of the scene.
          2.
          Original Goal Description: A armchair. 
          Enhanced Description: A sleek, modern blue armchair, upholstered in soft velvet, positioned prominently in a stylish living room. The chair is placed near a large floor-to-ceiling window, allowing natural light to highlight its deep blue hue. The room features minimalist decor with white walls, light wood flooring, and a few abstract art pieces on the walls. The armchair stands out as the main focal point, inviting comfort and relaxation.
          3.
          Original Goal Description: A desk. 
          Enhanced Description: A robust metal desk, with a weathered, matte surface, placed against a brick wall in an industrial-style office. The desk features exposed steel legs, and on its surface lies a sleek laptop, a coffee mug, and a few scattered papers. Overhead, a vintage filament bulb hangs from a chain, casting a warm glow over the scene. The surrounding decor includes minimalistic shelves and a large plant in the corner, yet the desk remains the focal point in the room’s urban, raw atmosphere.

        """
        },
        {"type": "text", "text": f"""
         Now, the Original Goal Description is "{text_prompt}", and the observation images are:
        """
        },
        {"type": "text", "text": f"observation1:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_images[0]}"}},
        {"type": "text", "text": f"observation2:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_images[2]}"}},
        {"type": "text", "text": f"""
         please follow the above requirements and examples to enhance this description, think step by step and give your analysis process and the final enhancement description following the format:
         **analysis process**: [your analysis process here]
         **enhancement description**: [your enhancement description here]
        """
        }]
        }]
        
    )
    
    return completion.choices[0].message.content


def imagenary_helper(client, text_prompt):
    completion = client.chat.completions.create(
      model="gpt-4o",
      timeout=500,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""
          You are an expert in refining and elaborating simple scene or object descriptions for use in text-to-image generation. 
          Your task is to take a given raw navigation target description—often just a short phrase mentioning an object or a simple scene—and enhance it by adding imaginative and contextual details. 
          It is crucial that the original object(s) remain the visual focal point. To achieve this:

          1.**Expand on the Original Description**: Add details about the object(s), their materials, colors, textures, and immediate surroundings.
          2.**Maintain Visual Dominance**: Ensure that any additional context or setting you provide does not overshadow the main elements described in the original text. 
          The main subjects should remain the most prominent features of the resulting image.
          3.**Emphasize the Main Objects**: Use language that clearly highlights the original objects, ensuring they stand out as the central focus.
          
          **Output Requirements:**
          1.Provide a refined, detailed description in English.
          2.Ensure the enhancements create a vivid, coherent, and inviting scene that supports the original description.
          3.Avoid overly intricate storytelling or elements that distract from the original object or scene.
          4.Please avoid overly long descriptions, limit descriptions to 70 words or less.
          
          **Guidelines for Creating Enhanced Descriptions:**

          1.**Detailing**: Incorporate sensory details such as color, texture, lighting, and reflections.
          2.**Contextual Elements**: Add subtle background elements or environmental details that complement but do not compete with the main subject.
          3.**Focus Phrasing**: Use language that naturally draws attention to the original object(s) (e.g., “centered,” “prominently placed,” “serving as the focal point”).
          4.**Balance**: Maintain a balance between enrichment and simplicity. The primary object or scene described in the original prompt should always dominate the final image.
        """
        },
        {"role": "user", "content": f"""

          **Examples**:
          1.
          Original Description:
          a TV screen above cabinets.
          Enhanced Description:
          A sleek, flat-screen TV mounted above a set of smooth, white cabinetry. The TV's reflective surface catches the soft glow of recessed lighting, 
          while the clean, minimalist cabinets provide a neat base that keeps the television as the primary visual anchor.
          2.
          Original Description:
          a marble island in kitchen.
          Enhanced Description:
          A polished white marble island centered in a modern kitchen. Delicate gray veining runs across its surface, subtly reflecting under the warm overhead lights. 
          Simple barstools and neat, neutral-toned countertops frame the island, ensuring it remains the kitchen's focal point.
          3.
          Original Description:
          a coffee mug on a desk.
          Enhanced Description:
          A sturdy ceramic coffee mug, ivory in color, resting on a clean, wooden desk. Soft light from a nearby window gently highlights the mug's curved handle and the steam rising from its freshly poured contents. 
          A simple laptop and a neatly arranged notepad stay in the background, making the mug stand out as the central feature.
          4.
          Original Description:
          lamp.
          Enhanced Description:
          A classic white table lamp with a slender stem and an elegant lampshade, standing on a minimalist nightstand. The soft glow from the lamp illuminates its graceful lines, 
          while the uncluttered background ensures the lamp remains the central feature of the scene.
          5.
          Original Description:
          a painting.
          Enhanced Description:
          A colorful, abstract painting mounted on a textured, red-brick wall. The painting's bold brushstrokes and vibrant hues stand out sharply against the wall's rough surface. 
          Soft track lighting above gently illuminates the artwork, drawing the eye directly to it.
                  
        """
        },
        {"role": "user", "content": f"""
         Now, the original description is "{text_prompt}", please follow the above requirements and examples to enhance this description, and directly output the enhanced description.
        """
        },
        
        ])
    
    return completion.choices[0].message.content

def imagenary_helper_long_text(client, text_prompt):
    goal_text_intrinsic, goal_text_extrinsic = text_prompt[0], text_prompt[1]
    completion = client.chat.completions.create(
      model="gpt-4o",
      timeout=500,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""
          You are an expert in refining and elaborating simple scene or object descriptions for use in text-to-image generation. 

          Your task is to reasonably merge the text description of the object instance and the text description of the surrounding environment, and output a complete description text to guide a text-to-image model to generate a photo of the object. It is crucial that the instance object(s) remain the visual focal point. To achieve these, you should:

          1.**Expand on the Original Description**: Based on the original object description and environment description, you should provides a more fine-grained description of materials, colors, and textures.

          2.**Maintain Visual Dominance**: Ensure that any additional context or setting you provide does not overshadow the main elements described in the original text. the main object should remain the most prominent features of the resulting image.

          **Output Requirements:**

          1.Provide a refined, detailed description in English.

          2.Ensure the enhancements create a vivid, coherent, and inviting scene that supports the original description.

          3.Avoid overly intricate storytelling or elements that distract from the original object or scene.

          4.The extended description must follow the original description and not contradict it, such as color, material, and appearance.

          4.Please avoid overly long descriptions, limit descriptions to 70 words or less.
                    
          **Guidelines for Creating Enhanced Descriptions:**

          1.**Detailing**: Incorporate sensory details such as color, texture, lighting, and reflections.

          2.**Contextual Elements**: Add subtle background elements or environmental details that complement but do not compete with the main subject.

          3.**Focus Phrasing**: Use language that naturally draws attention to the original object(s) (e.g., “centered,” “prominently placed,” “serving as the focal point”).

          4.**Balance**: Maintain a balance between enrichment and simplicity. The primary object or scene described in the original prompt should always dominate the final image.
        """
        },
        {"role": "user", "content": f"""

          **Examples**:

          "intrinsic_attributes": "According to the image description, this chair is an old wooden structure with blue seat cushions. Its specific material and color cannot be determined because there are no other details provided about its body or frame."

          "extrinsic_attributes": "There are no other objects around the chair in this image. Only a very simple environment is shown, with only one object: an old wooden table and four blue chairs."

          Enhanced Description Output:
          A weathered wooden chair with a timeworn frame and soft blue cushions sits prominently at the center. The rich grain of the wood contrasts with the smooth fabric, which shows subtle signs of wear. The surroundings are minimal, featuring a matching wooden table and four identical blue chairs, creating a serene and uncomplicated environment where the chair remains the clear focal point. The warm, natural light highlights the texture of the wood and fabric.
                            
        """
        },
        {"role": "user", "content": f"""
         Now, the "intrinsic_attributes" is "{goal_text_intrinsic}", the "extrinsic_attributes" is "{goal_text_extrinsic}", please follow the above requirements and examples to enhance this description, and directly output the enhanced description.
        """
        },
        
        ])
    
    return completion.choices[0].message.content



def long_memory_localized(client, text_prompt, long_memory):
    
      completion = client.chat.completions.create(
      model="gpt-4o",
      timeout=500,
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": """
        You are an LLM Agent with a specific goal: given a textual description of a navigation target (e.g., “A marble island in a kitchen.”) 
        and a memory list containing instances of detected objects, each with a label, a 3D location (three numerical coordinates), and a confidence score, 
        you need to determine the most suitable memory instance to fulfill the navigation request.

        **Your memory data structure:**
        The memory is a list of objects in the environment, where each object is represented as a JSON-like structure of the following form:
        {
        "label": "<string>",
        "loc": [<float or int>, <float or int>, <float or int>],
        "confidence": <float>
        }
        label: A textual label describing the object (e.g., “a tv”, “a kitchen island”).
        loc: A three-element array representing the coordinates of that object in the environment.
        confidence: A floating-point value indicating how confident the system is in identifying this object as described by label.
        
        **Your task:**

        1.Understand the target description: You will be given a textual goal description of a navigation target, such as “A marble island in a kitchen.” 
        Your first step is to interpret this description and deduce which object label from the memory best matches it semantically. 
        For example, if the target is “A marble island in a kitchen,” and you have memory instances labeled “a kitchen island” or “a marble island,” 
        you should identify that these instances correspond to the target description. Consider synonyms and close matches. If no exact label is found, 
        choose the label that is most semantically similar to the target description. For instance, if the target mentions “a marble island” and the memory only has “a kitchen island,” 
        you should still select the “a kitchen island” label as it is likely the intended object.

        2.Identify the relevant instances: Once you have determined the best matching label, filter the memory list to only those instances whose label matches (or closely matches) that label.

        3.Evaluate confidence and consolidate duplicates: Among these filtered instances, consider that multiple memory entries may actually represent the same object, 
        possibly due to partial overlaps or multiple detections.

        - Look at their loc coordinates. If multiple instances with the same label have very close or nearly identical coordinates, treat them as the same object.
        - Determine which set of coordinates (if there are multiple distinct sets) is the most reliable representation of the object. Reliability is judged primarily by the highest confidence value. If multiple instances cluster together with similar locations, select the one with the highest confidence or, if confidence is similar, the one that best aligns with the object as described.

        4.Select the final loc: After you have grouped instances and decided which group best represents the target object, output the coordinates (loc) of the best match. 
        If multiple objects (>=3 items) match the description equally well, choose the three coordinates (loc) with the highest confidence.

        5.Produce a final answer: Return the selected location coordinates as the final answer, (important!!) must be in the format '**Result**: (Nav Loc 1: [...], Nav Loc 2: [...], Nav Loc 3: [...])' or '**Result**: (Nav Loc: Unable to find)'.
        
        **Important details:**

        - Always provide reasoning internally (you may do it in hidden scratchpads if available) before giving the final result. 
        The final user-visible answer should be concise and directly address the task.
        - If no objects are found that are semantically relevant to the target description, explicitly indicate that no suitable object was found.
        - Follow these steps for every input you receive. 
        """
        },
        
        {"role": "user", "content": f"navigation target:{text_prompt}"},
        {"role": "user", "content": f"memory:{long_memory}"},
        
        {"role": "user", "content": "Now please start thinking one step at a time and then Briefly tell me the target location I need to go to and return as '**Result**: (Nav Loc 1: [...], Nav Loc 2: [...], Nav Loc 3: [...])' format, If there is no suitable target in memory, return as '**Result**: (Nav Loc: Unable to find)"},
        
      ])


      return completion.choices[0].message.content
    
def image_to_base64(images: Image.Image, fmt="JPEG") -> str:

    base64_images = []
    for img in images:
      output_buffer = BytesIO()
      img.save(output_buffer, format=fmt)
      byte_data = output_buffer.getvalue()
      base64_str = base64.b64encode(byte_data).decode('utf-8')
      base64_images.append(base64_str)
    
    return base64_images
    
# def succeed_determine(client, text_prompt, obs):
#       base64_image = image_to_base64(obs)
      
#       completion = client.chat.completions.create(
#       model="gpt-4o",
#       messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": """
#         You are an AI assistant tasked with assessing whether a navigation task has been successfully completed. You will be given:
#         1.A observation image (the image observed at the end of navigation).
#         2.A text description of the target object the navigation aimed to locate.
#         Your goal is to determine:
#         1.Explanation: Provide a concise explanation showing how you arrived at your conclusion, referencing any matching or non-matching details between the final observation and the target description.
#         2.Success or not: Does the final observation indicate that the agent has found the target object?
        
#         **Important Requirements**
#         Before your success verdict, provide an explanation on a line, starting with Explanation: and then your reasoning.
#         If the final observation includes the target object based on the textual description, respond with exactly Success: yes.
#         If the final observation does not include the target object, respond with exactly Success: no.
        
#         Do not add extra lines or deviate from the required format.
#         **Format of Your Answer**
#         First line: Explanation: [Your explanation here]
#         Second line: Success: yes OR Success: no
#         """
#         },
        
        
#         {"role": "user", 
#          "content": [
#            {
#               "type": "text",
#               "text": "observation image:"},
#            {
#               "type": "image_url",
#               "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},},
#          ]
#          },
#         {"role": "user", "content": f"target description:{text_prompt}"},
        
#         {"role": "user", "content": "Now please start thinking step by step"},
        
#       ])


#       return completion.choices[0].message.content

def succeed_determine(client, text_prompt, obss):
  base64_images = image_to_base64(obss)
    
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [ 
      
      {
        "type": "text",  "text":                        
        """
        You will be provided with 2 navigation observation images from from different perspectives of the agent, and a textual description of the navigation goal. Please follow the steps below to determine whether the current navigation task is successful:
        1.Determine Target Presence: Analyze the provided images one by one, to ascertain whether the navigation goal is present in these images. This means evaluating if the agent has arrived near the target location.
        2. Output Format:
        First Line: Success: yes OR Success: no
        Second Line: Give your analysis results in detail
        Examples
        '''
        Success: yes
        [analysis results]
        '''
        or
        '''
        Success: no
        [analysis results]
        '''

        Please analyze according to the above requirements and respond strictly in the specified format.
        """
      },
      {
        "type": "text",  "text": f"target description: {text_prompt}"   
      } 
                                
    ]
    },]
  
  for idx, base64_image in enumerate(base64_images):
      messages[1]["content"].extend([
          {"type": "text", "text": f"observation view {idx}:"},
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
      ])
  
  messages[1]["content"].extend([
          {"type": "text", "text": "Now please start thinking step by step"},
      ])    
    
  completion = client.chat.completions.create(
  model="gpt-4o",
  timeout=500,
  messages=messages
  )


  return completion.choices[0].message.content



def succeed_determine_singleview(client, text_prompt, obss):
  base64_images = image_to_base64(obss)
    
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [ 
      
      {
        "type": "text",  "text":                        
        """
        You will be provided with a navigation observation image from a robot, and a textual description of the navigation goal. Please follow the steps below to determine whether the current navigation task is successful:
        1.Determine Target Presence: Analyze the provided images to ascertain whether the navigation goal is present in these images And close enough (within 2 meters). This means evaluating if the robot has arrived near the target location. Be careful not to misclassify the similar categories (e.g. sofas and chairs are easily confused)
        2.Determine whether need to move forward. If you have found the target according to the step 1, you need to further determine whether you need to move forward a small step to get closer to the target object. If need to, answer 'need forward: yes'. If you think you are close enough (within 1m), answer 'need forward: no'.
        3. Output Format:
        First Line: Success: yes OR Success: no
        Second Line (only when 'success: yes'): need forward: yes OR need forward: no
        Third Line: Give your analysis results in detail
        Examples
        '''
        Success: yes
        need forward: yes
        [analysis results]
        '''
        or
        '''
        Success: yes
        need forward: no
        [analysis results]
        '''
        or
        '''
        Success: no
        [analysis results]
        '''

        Please analyze according to the above requirements and respond strictly in the specified format.
        """
      },
      {
        "type": "text",  "text": f"target description: {text_prompt}"   
      } 
                                
    ]
    },]
  
  for idx, base64_image in enumerate(base64_images):
      messages[1]["content"].extend([
          {"type": "text", "text": f"observation view:"},
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
      ])
  
  messages[1]["content"].extend([
          {"type": "text", "text": "Now please start thinking step by step"},
      ])    
    
  completion = client.chat.completions.create(
  model="gpt-4o",
  timeout=500,
  messages=messages
  )


  return completion.choices[0].message.content



def succeed_determine_singleview_with_imggoal(client, img_prompt, obss):
  obss = [obs.resize((512, 512)) for obs in obss]
  img_prompt = img_prompt.resize((512, 512))
  base64_images = image_to_base64(obss)
  base64_images_goal = image_to_base64([img_prompt])
    
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [ 
      
      {
        "type": "text",  "text":                        
        """
        You will get an image of the navigation target and an image of the current observation from the robot. 
        1. Analyze the main instance objects contained in the two images, especially the closest objects. Think about what objects they are? What appearance features do they have?
        2. Compare the two images, combined with the analysis of the step 1, to determine whether the current robot has reached the vicinity of the target. This means that the current observation image is taken at the location of the navigation target image.
        3. Determine whether need to move forward. If you think the robot has reached the target, you need to further determine whether it needs to move forward a small step to get closer to the target object.
        Note: The viewpoints of the two images are usually different. You need to judge carefully to avoid misjudgment.
        Output Format:
        First Line: Success: yes OR Success: no
        Second Line (only when 'success: yes'): need forward: yes OR need forward: no
        Third Line: Give your analysis results in detail
        Examples
        '''
        Success: yes
        need forward: yes
        [analysis results]
        '''
        or
        '''
        Success: yes
        need forward: no
        [analysis results]
        '''
        or
        '''
        Success: no
        [analysis results]
        '''

        Please analyze according to the above requirements and respond strictly in the specified format.
        """
      },
      {
         "type": "text", "text": f"navigation goal image:"
      },
      {
         "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_images_goal[0]}"}
      },
      {
         "type": "text", "text": f"current observation image:"
      },
      {
         "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_images[0]}"}
      },  
      {
         "type": "text", "text": "Now please start analysing."
      },                          
    
    ],
    }
    ]
    
  completion = client.chat.completions.create(
  model="gpt-4o",
  timeout=500,
  messages=messages
  )


  return completion.choices[0].message.content



def touching_helper(client, text_prompt, obss, model=None, processor=None):
  base64_image = image_to_base64(obss)[0]
  
  messages = [
    {"role": "user", 
     "content": 
     [ 
      {
        "type": "text",  "text":                        
        """
        Suppose you are an agent performing a navigation task, and you need to make yourself as close to a specified target as possible. Now that you have reached the vicinity of the target, I will provide you with a description of the target object you need to reach and the current observation image. Please analyze and make decisions according to the following steps:
        1. Direction judgment:
        Based on the observation image, is the target object already in your field of vision? If it appears, in which direction is it located? For example: straight ahead/left front/right front. If it does not appear, determine in which direction it is most likely to appear.
        2. Strategy decision:
        Based on the direction you analyzed, if there is still a certain distance from the target, what should be your next best movement strategy? You have the following options: ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'finish_task']. Please note that you only need to consider one step of strategy. 
        If you think you are close enough (the distance is less than 1m), you should choose the 'finish_task' strategy.
        3. Output format:
        The final answer needs to be output in a strict format. For example: **Strategy**: 'xxx' (xxx is the strategy you choose).

        Please analyze according to the above requirements and respond strictly in the specified format.
        """
      },
      {
        "type": "text",  "text": f"target description: {text_prompt}"   
      },
      {
         "type": "text", "text": f"observation view:"   
      }, 
      {
         "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
      },
      {
         "type": "text", "text": "Now please start thinking step by step"
      },
                                
    ],
    } 
    ]

  # text = processor.apply_chat_template(
  #     messages, tokenize=False, add_generation_prompt=True
  # )
  # image_inputs, video_inputs = process_vision_info(messages)
  # inputs = processor(
  #     text=[text],
  #     images=image_inputs,
  #     videos=video_inputs,
  #     padding=True,
  #     return_tensors="pt",
  # )
  # inputs = inputs.to("cuda")

  # # Inference: Generation of the output
  # generated_ids = model.generate(**inputs, max_new_tokens=128)
  # generated_ids_trimmed = [
  #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
  # ]
  # output_text = processor.batch_decode(
  #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
  # )
  
  # return output_text[0]



  completion = client.chat.completions.create(
  model="gpt-4o",
  timeout=500,
  messages=messages
  )

  return completion.choices[0].message.content




def vln_subgoal_planner_with_obs(client, text_prompt):

  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [ 
      
      {
        "type": "text",  "text":                        
        """
        You will get a text instruction for long-distance navigation in an indoor environment. 
        Now, your task is to decompose the text instruction into reasonable and clear sub-task goals to help the agent complete the complex navigation task step by step. All sub-tasks need to be expressed in the form of "{move to ...}", where the target in {...} can be a word or description of an object, or a description of a room area. 
        Next, I will give you some examples:

        **Text prompt:**
        "Walk into the hallway and at the top of the stairs turn left and walk into the bedroom. Walk past the right side of the bed and turn right into the closet and all the way through to the bathroom. Stop in front of the toilet in the bathroom."
        
        **Response:**
        1. Move to the {stairs at the end of the hallway}
        2. Move to the {bed in the bedroom}
        3. Move to the {closet}
        4. Move to the {toilet in the bathroom}

        **Text prompt:**
        "Go to the wooden stairs. Go up the stairs and go between the couch and the table. Walk into the house through the sliding glass door. Go to the television. Go to the refrigerator. Go to the front of the toaster and stop"
        
        **Response:**
        1. Move to the {wooden stairs}
        2. Move to the {area between a couch and a table}
        3. Move to the {sliding glass door}
        4. Move to the {television}
        5. Move to the {refrigerator}
        6. Move to the {toaster}

        Now, please planning the following text prompt into sub-goals and respond strictly in the specified format. Do not include any other information.
        """
      },
      {
         "type": "text", "text": f"**Text prompt:** {text_prompt}"
      },                    
    
    ],
    }
    ]
  
    
  completion = client.chat.completions.create(
  model="gpt-4o",
  timeout=500,
  messages=messages
  )


  return completion.choices[0].message.content



def vln_subgoal_planner_no_object(client, text_prompt):
    
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [ 
      
      {
        "type": "text",  "text":                        
        """
        You will get a text instruction for long-distance navigation in an indoor environment. 
        Now, your task is to decompose the text instruction into reasonable and clear sub-task goals to help the agent complete the complex navigation task step by step. 
        Next, I will give you some examples, the sub-task must in the format of "{...}":

        **Text prompt:**
        "Walk into the hallway and at the top of the stairs turn left and walk into the bedroom. Walk past the right side of the bed and turn right into the closet and all the way through to the bathroom. Stop in front of the toilet in the bathroom."
        
        **Response:**
        1. {walk into the hallway and at the top of the stairs.}
        2. {turn left and walk into the bedroom.}
        3. {walk past the right side of the bed}
        4. {turn right into the closet and all the way through to the bathroom.}
        5. {Stop in front of the toilet in the bathroom}

        **Text prompt:**
        "Go to the wooden stairs. Go up the stairs and go between the couch and the table. Walk into the house through the sliding glass door. Go to the television. Go to the refrigerator. Go to the front of the toaster and stop"
        
        **Response:**
        1. {go to the wooden stairs}
        2. {go up the stairs}
        3. {go between the couch and the table}
        4. {walk into the house through the sliding glass door}
        5. {go to the television}
        6. {go to the refrigerator}
        7. {go to the front of the toaster and stop}

        Now, please planning the following text prompt into sub-goals and respond strictly in the specified format. Do not include any other information.
        """
      },
      {
         "type": "text", "text": f"**Text prompt:** {text_prompt}"
      },
                             
    
    ],
    }
    ]

  completion = client.chat.completions.create(
  model="gpt-4o",
  timeout=500,
  messages=messages
  )


  return completion.choices[0].message.content


def vln_anchor_planner(client, text_prompt, obss):
  base64_image = image_to_base64(obss)
  
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [ 
      
      {
        "type": "text",  "text":                        
        """
        You will act as an intelligent assistant to complete a scene navigation task. You need to analyze the provided observation images and determine the most appropriate direction of movement based on the given navigation instructions. Once the direction is determined, you need to use the corresponding image to predict and describe any important objects or environmental features that you will encounter at the destination. Your description should use natural language and provide detailed descriptions of the object's external features (texture, shape, etc.). If no obvious object is found, you should describe the surrounding environment and any significant structures or markers within it.

        Here are the task breakdown steps you need to follow:

        Direction Determination: Based on the navigation instruction, you need to choose the correct directional image from the provided images. This direction should be derived from the images provided and must be a logical response to the instruction.

        Object Analysis: After determining the correct directional image, analyze the image to identify what "anchor objects" you will encounter at the destination. Anchor objects represent the physical objects you will encounter when following the abstract instruction. An anchor object can be any object, such as a piece of furniture, a notable landmark, or any other clearly identifiable structure. You should provide a detailed natural language description of the anchor object, including its appearance, such as shape, color, texture, and any other notable features. If no obvious object is found in the selected direction, try to describe the appearance of the destination. This might include the terrain, nearby notable structures, or any significant features.

        Output Response: Finally, you need to return your analysis results and provide the anchor object description.

        Here is an example. You must strictly follow the format of the following response:

        Instruction: "Walk straight through the bathroom and exit."

        Your Response should be in the following format:
        
        Analysis: (your analysis here)

        Anchor Object: e.g., A large marble sink with gold-trimmed faucets. The sink is rectangular, with polished white marble and faint gray veining running across its surface. Above the sink, there's a framed mirror with an ornate golden frame.

        Now, please start your analysis based on the following input information:
        """
      },
      {
         "type": "text", "text": f"**Instruction:** {text_prompt}"
      },
      {
         "type": "text", "text": f"**Observation images:**"
      }
    
    ],
    }
    ]
  
  for idx, base64_image in enumerate(base64_image):
      messages[1]["content"].extend([
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
      ])
    
  completion = client.chat.completions.create(
  model="gpt-4o",
  timeout=500,
  messages=messages
  )


  return completion.choices[0].message.content





def vln_anchor_planner_v2(client, text_prompt, obss):
  base64_image = image_to_base64(obss)
  
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [ 
      
      {
        "type": "text",  "text":                        
        """
        You will act as an agent to complete the task of assisting scene navigation. I will provide you with a navigation instruction and a series of current observations. The navigation instruction requires the agent to move from the current position to an "object" not far away. However, the current description of the object is rough and is only a simple category description. 

        Now, your task is to analyze the current observation and then describe the characteristics of the target object in as much detail as possible, including fine-grained elements such as appearance, shape, and texture. You may encounter two situations: 

        1. The target object is within the field of view of the observed image, which indicates that you can make a fine-grained description based on the observation content. 

        2. The target object does not exist in the field of view of the observed image. At this time, you need to associate and infer the possible characteristics of the target object as much as possible based on the indoor environment and surrounding information in the observed image, and give a description.       

        Here is an example. You need to output the description directly without any other analysis or additional responses:

        Input Instruction: 

        "Move to the marble sink in the bathroom."

        Your Response should like:

        "A large marble sink with gold-trimmed faucets. The sink is rectangular, with polished white marble and faint gray veining running across its surface. Above the sink, there's a framed mirror with an ornate golden frame."

        Now, please start your analysis based on the following input information:
        """
      },
      {
         "type": "text", "text": f"**Instruction:** {text_prompt}"
      },
      {
         "type": "text", "text": f"**Observation images:**"
      }
    
    ],
    }
    ]
  
  for idx, base64_image in enumerate(base64_image):
      messages[1]["content"].extend([
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
      ])
    
  completion = client.chat.completions.create(
  model="o3",
  timeout=500,
  messages=messages
  )


  return completion.choices[0].message.content



def EQA_generate_anchor_object(client, text_prompt):
    messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [ 
        
        {
          "type": "text",  "text":                        
          """
          You will act as an agent to complete the task of assisting scene embodied question answering. I will provide you with a question about the scene. In order to answer this question, we must first navigate to the vicinity of the instance involved in the question.

          Now, your task is to analyze and determine the description of the target instance you need to move to based on the current question, which can include some necessary spatial context, such as what type of room this target instance is in and what clear objects exist around it. If you think it is difficult to infer the exact target instance for the type of question provided, please output "We need to go around and check"

          I will provide you with some examples. You need to output the description directly without including other analysis and additional response content.

          Example 1:

          Question: What is the white object on the wall above the TV?

          Response: Now, we need to go to {A TV mounted on the wall.}

          Example 2:

          Question: What is in between the two picture frames on the blue wall in the living room?

          Response: Now, we need to go to {A blue wall in the living room with two picture frames on it.}

          Example 3:

          Question: What should I do to cool down?

          Response: We need to go around and check.

          Your Turn:
          """
        },
        {
          "type": "text", "text": f"Question: {text_prompt} Response:"
        }
      
      ],
      }
      ]
    
      
    completion = client.chat.completions.create(
    model="o3-mini",
    timeout=500,
    messages=messages
    )


    return completion.choices[0].message.content


def EQA_Answer_o3(client, text_prompt, obss):
  base64_image = image_to_base64(obss)
  
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [ 
      
      {
        "type": "text",  "text":                        
        f"""
        You are an intelligent question-answering robot. I will ask you some questions about indoor spaces, and you must give answers.

        You will see a set of images collected from the same location. The location of these images is related to the question, and you must carefully reason from them to get the correct answer. If you think there is not enough information to get the answer at that location, please try to guess a close and most likely answer.

        Based on the user query, you must output "text" to answer the question asked by the user. Here are some examples:

        Example1:
        Question: What is to the left of the mirror?
        Answer: A plant in a tall vase.

        Example2:
        Question: Where is the mirror?
        Answer: Next to the staircase above the dark brown cabinet.

        Your Turn:
        Question: {text_prompt}

        """
      },
      {
         "type": "text", "text": f"**Observation images:**"
      }
    
    ],
    }
    ]
  
  for idx, base64_image in enumerate(base64_image):
      messages[1]["content"].extend([
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
      ])
    
  completion = client.chat.completions.create(
  model="o3-mini",
  timeout=500,
  messages=messages
  )


  return completion.choices[0].message.content

def EQA_Answer_4o(client, text_prompt, obss):
  base64_image = image_to_base64(obss)
  
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [ 
      
      {
        "type": "text",  "text":                        
        f"""
        You are an intelligent question-answering robot. I will ask you some questions about indoor spaces, and you must give answers.

        You will see a set of images collected from the same space. These observation images is related to the question, and you must carefully reason from them to get the correct answer. Please do not respond with "I can't answer that because I didn't see..." If you think the images are not accurate enough to get an answer, please try to guess the most likely answer.

        Based on the user query, you must output "text" to answer the question asked by the user. Here are some examples:

        Example1:
        Question: What is to the left of the mirror?
        Answer: A plant in a tall vase.

        Example2:
        Question: Where is the mirror?
        Answer: Next to the staircase above the dark brown cabinet.

        Your Turn:
        Question: {text_prompt}

        """
      },
      {
         "type": "text", "text": f"**Observation images:**"
      }
    
    ],
    }
    ]
  
  for idx, base64_image in enumerate(base64_image):
      messages[1]["content"].extend([
          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
      ])
    
  completion = client.chat.completions.create(
  model="gpt-4o",
  timeout=500,
  messages=messages
  )


  return completion.choices[0].message.content