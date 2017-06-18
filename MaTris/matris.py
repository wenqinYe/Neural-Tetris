#!/usr/bin/env python
import pygame
from pygame import Rect, Surface
import random
import os
import kezmenu
import time

import cPickle as pickle

from tetrominoes import list_of_tetrominoes
from tetrominoes import rotate

import copy

from scores import load_score, write_score

class BrokenMatrixException(Exception):
    pass

def get_sound(filename):
    return pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "resources", filename))

BGCOLOR = (15, 15, 20)
BORDERCOLOR = (140, 140, 140)

BLOCKSIZE = 30
BORDERWIDTH = 10

MATRIS_OFFSET = 20

WIDTH = 700
HEIGHT = 20*BLOCKSIZE + BORDERWIDTH*2 + MATRIS_OFFSET*2

MATRIX_WIDTH = 10
MATRIX_HEIGHT = 22
VISIBLE_MATRIX_HEIGHT = MATRIX_HEIGHT - 2
import numpy as np

memory = []

def track_move(current_state, move, reward, new_state, rot_before, rot_after, current_tetromino, next_tetromino):
    #file_path = '/Users/wenqin/Documents/GitHub/grade-12-assignments-wenqinYe/Culminating/memory_v2_bad_play.pkl'
    #save_object((current_state, move, reward, new_state, rot_before, rot_after, current_tetromino.shape, next_tetromino.shape), file_path)
    pass
class Matris(object):
    def __init__(self, size=(MATRIX_WIDTH, MATRIX_HEIGHT), blocksize=BLOCKSIZE):
        self.size = {'width': size[0], 'height': size[1]}
        self.blocksize = blocksize
        self.surface = Surface((self.size['width']  * self.blocksize,
                                (self.size['height']-2) * self.blocksize))


        self.matrix = dict()
        for y in range(self.size['height']):
            for x in range(self.size['width']):
                self.matrix[(y,x)] = None
                
        self.matrix_pretransform = copy.deepcopy(self.matrix)
                
        self.processing_matrix = np.zeros((MATRIX_HEIGHT, MATRIX_WIDTH))

        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.set_tetrominoes()
        self.tetromino_rotation = 0
        self.downwards_timer = 0
        self.base_downwards_speed = 0.4 # Move down every 400 ms

        self.movement_keys = {'left': 0, 'right': 0}
        self.movement_keys_speed = 0.05
        self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.level = 1
        self.score = 0
        self.lines = 0

        self.combo = 1 # Combo will increase when you clear lines with several tetrominos in a row
        
        self.paused = False
        self.gameover = False

        self.highscore = load_score()
        self.played_highscorebeaten_sound = False
        
        # self.levelup_sound  = get_sound("levelup.wav")
        # self.gameover_sound = get_sound("gameover.wav")
        # self.linescleared_sound = get_sound("linecleared.wav")
        # self.highscorebeaten_sound = get_sound("highscorebeaten.wav")
        
        
        """
        Instance variables for the AI
        """
        self.tetromino_rotation_ai = 0
        self.tetromino_position_ai = None
        
        
    """
    AI Code
    """
    def execute_moves(self, moves):
        delta_x = moves[0]
        rotation = moves[1]
        
        x_movement = delta_x - self.tetromino_position[1]
        move_type = 'right'
        if x_movement < 0:
            move_type = 'left'
        for i in range(np.abs(x_movement)):
            self.request_movement(move_type)
        self.request_rotation(rotation)
        
        self.hard_drop()
        
    def execute_move(self, move):
        move_dict = {
            "hard_drop" : lambda: self.hard_drop(),
            "rotate0": lambda: self.request_rotation(rotation_num=0),
            "rotate1": lambda: self.request_rotation(rotation_num=1),
            "rotate2": lambda: self.request_rotation(rotation_num=2),
            "rotate3": lambda: self.request_rotation(rotation_num=3),
            "left": lambda: self.request_movement("left"),
            "right": lambda: self.request_movement("right")
        }
        
        move_dict[move]()
        
    def execute_move_index(self, index):
        moves = ["hard_drop", "left", "right", "rotate0", "rotate1", "rotate2", "rotate3"]
        self.execute_move(moves[index])
    def get_processing_matrix(self):
        for y in range(self.size['height']):
            for x in range(self.size['width']):
                if self.matrix[(y, x)] != None:
                    self.processing_matrix[y][x] = 1
        return self.processing_matrix
                    
    def dict_to_matrix(self, dict_matrix):
        """
        Converts a tetris dictionary board representation to a
        matrix representation
        """
        matrix = np.zeros((self.size['height'], self.size['width']))
        for y in range(self.size['height']):
            for x in range(self.size['width']):
                if dict_matrix[(y, x)] != None:
                    matrix[y][x] = 1
        return matrix   
        
    def generate_next_states(self, matrix, tetromino):
        num_rotations = 4
        position = [0, 0] #y=0, x=0
        end_states = []
        for rotation in range(num_rotations):
            rotated_shape = copy.copy(self.rotated(rotation, tetromino=tetromino))
            for delta_x in range(8):
                position = copy.copy(position)
                """
                If matrix is None, blend will do shape blending on the current matrix.
                
                Checks to see if block can be moved into x-position
                """
                if(not self.blend(rotated_shape, (position[0], position[1]+delta_x), matrix=matrix)):
                    """
                    Piece can't move in this x-position
                    """
                    continue
                    
                """
                Simulates hard dropping
                """
                delta_y = 0
                while(self.blend(rotated_shape, (position[0]+delta_y, position[1]+delta_x), matrix=matrix)):
                    delta_y += 1
                delta_y -= 1

                end_state = self.blend(rotated_shape, (position[0]+delta_y, position[1]+delta_x), matrix=matrix)
                """
                There is a bug where this may be false because the block can't move down at all.
                """
                end_states.append([end_state, (delta_x, rotation)])
        return end_states
        
        
    def check_rotation(self, rotation_num=0):
        """
        Gets the rotated version of the shape and sees if
        it fits inside the matrix.
        
        If it does then set that as the rotation and position
        of the shape.
        
        There are 4 possible rotation numbers: 0, 1, 2, 3
        """
        rotation = rotation_num
        shape = self.rotated(rotation)

        y, x = self.tetromino_position_ai

        position = (self.fits_in_matrix(shape, (y, x)) or
                    self.fits_in_matrix(shape, (y, x+1)) or
                    self.fits_in_matrix(shape, (y, x-1)) or
                    self.fits_in_matrix(shape, (y, x+2)) or
                    self.fits_in_matrix(shape, (y, x-2)))
        # ^ Thats how wall-kick is implemented

        if position and self.blend(shape, position):
            return True
        else:
            return False
                     

    """
    Original code
    """
    def set_tetrominoes(self):
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = random.choice(list_of_tetrominoes)
        self.surface_of_next_tetromino = self.construct_surface_of_next_tetromino()
        self.tetromino_position = (0,4) if len(self.current_tetromino.shape) == 2 else (0, 3)
        self.tetromino_rotation = 0
        self.tetromino_block = self.block(self.current_tetromino.color)
        self.shadow_block = self.block(self.current_tetromino.color, shadow=True)

    
    def hard_drop(self):
        amount = 0
        while self.request_movement('down'):
            amount += 1

        self.lock_tetromino()
        
        #self.score += 10*amount
        

    def update(self, timepassed):
        pressed = lambda key: event.type == pygame.KEYDOWN and event.key == key
        unpressed = lambda key: event.type == pygame.KEYUP and event.key == key
        
        events = pygame.event.get()
        
        for event in events:
            if pressed(pygame.K_p):
                self.surface.fill((0,0,0))
                self.paused = not self.paused
            elif event.type == pygame.QUIT:
                self.prepare_and_execute_gameover(playsound=False)
                exit()
            elif pressed(pygame.K_ESCAPE):
                self.prepare_and_execute_gameover(playsound=False)

        if self.paused:
            return
        
        move = None
        rot_before = None
        rot_after = None
        current_tetromino = self.current_tetromino
        next_tetromino = None
        original_score = self.score
        original_state = self.matrix
        
        for event in events:
            if pressed(pygame.K_SPACE):
                self.hard_drop()
                move = 'hard_drop'
            elif pressed(pygame.K_UP) or pressed(pygame.K_w):
                move = 'request_rotation'
                rot_before = self.tetromino_rotation
                self.request_rotation()
                rot_after = self.tetromino_rotation
            elif pressed(pygame.K_LEFT) or pressed(pygame.K_a):
                self.request_movement('left')
                self.movement_keys['left'] = 1
                move = 'left'
                
            elif pressed(pygame.K_RIGHT) or pressed(pygame.K_d):
                self.request_movement('right')
                self.movement_keys['right'] = 1
                move = 'right'

            elif unpressed(pygame.K_LEFT) or unpressed(pygame.K_a):
                self.movement_keys['left'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed)*2
            elif unpressed(pygame.K_RIGHT) or unpressed(pygame.K_d):
                self.movement_keys['right'] = 0
                self.movement_keys_timer = (-self.movement_keys_speed)*2

        self.downwards_speed = self.base_downwards_speed ** (1 + self.level/10.)

        self.downwards_timer += timepassed
        downwards_speed = self.downwards_speed*0.10 if any([pygame.key.get_pressed()[pygame.K_DOWN],
                                                            pygame.key.get_pressed()[pygame.K_s]])   else self.downwards_speed
        
        """
        This checks to ensure that the matris is not false before moving things down
        It will be false if an end game state has been reached
        """
        if self.matrix is not False:
            if self.downwards_timer > downwards_speed:
                if not self.request_movement('down'):
                    self.lock_tetromino()
                self.downwards_timer %= downwards_speed
            reward = self.score - original_score

        new_state = self.matrix
        next_tetromino = self.current_tetromino
        if move is not None:
            track_move(original_state, move, reward, new_state, rot_before, rot_after, current_tetromino, next_tetromino)
        
        if any(self.movement_keys.values()):
            self.movement_keys_timer += timepassed
        if self.movement_keys_timer > self.movement_keys_speed:
            result = self.request_movement('right' if self.movement_keys['right'] else 'left')
            self.movement_keys_timer %= self.movement_keys_speed
        
        if self.matrix is not False:
            with_shadow = self.place_shadow()

        try:
            with_tetromino = self.blend(self.rotated(), allow_failure=False, matrix=with_shadow)
        except BrokenMatrixException:
            self.prepare_and_execute_gameover()
            return

        for y in range(self.size['height']):
            for x in range(self.size['width']):
                #                                       I hide the 2 first rows by drawing them outside of the surface
                block_location = Rect(x*self.blocksize, (y*self.blocksize - 2*self.blocksize), self.blocksize, self.blocksize)
                if with_tetromino[(y,x)] is None:
                    self.surface.fill(BGCOLOR, block_location)
                else:
                    if with_tetromino[(y,x)][0] == 'shadow':
                        self.surface.fill(BGCOLOR, block_location)
                    
                    self.surface.blit(with_tetromino[(y,x)][1], block_location)
                    
    def prepare_and_execute_gameover(self, playsound=False):
        # if playsound:
        #     self.gameover_sound.play()
        write_score(self.score)
        self.gameover = True

    def place_shadow(self):
        posY, posX = self.tetromino_position
        while self.blend(position=(posY, posX)):
            posY += 1

        position = (posY-1, posX)

        return self.blend(position=position, block=self.shadow_block, shadow=True) or self.matrix
        # If the blend isn't successful just return the old matrix. The blend will fail later in self.update, it's game over.

    def fits_in_matrix(self, shape, position):
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if self.matrix.get((y, x), False) is False and shape[y-posY][x-posX]: # outside matrix
                    return False

        return position
                    

    def request_rotation(self, rotation_num=None):
        rotation = (self.tetromino_rotation + 1) % 4
        if rotation_num is not None:
            rotation = rotation_num
        shape = self.rotated(rotation)

        y, x = self.tetromino_position

        position = (self.fits_in_matrix(shape, (y, x)) or
                    self.fits_in_matrix(shape, (y, x+1)) or
                    self.fits_in_matrix(shape, (y, x-1)) or
                    self.fits_in_matrix(shape, (y, x+2)) or
                    self.fits_in_matrix(shape, (y, x-2)))
        # ^ Thats how wall-kick is implemented

        if position and self.blend(shape, position):
            self.tetromino_rotation = rotation
            self.tetromino_position = position
            return self.tetromino_rotation
        else:
            return False
            
    def request_movement(self, direction):
        posY, posX = self.tetromino_position
        if direction == 'left' and self.blend(position=(posY, posX-1)):
            self.tetromino_position = (posY, posX-1)
            return self.tetromino_position
        elif direction == 'right' and self.blend(position=(posY, posX+1)):
            self.tetromino_position = (posY, posX+1)
            return self.tetromino_position
        elif direction == 'up' and self.blend(position=(posY-1, posX)):
            self.tetromino_position = (posY-1, posX)
            return self.tetromino_position
        elif direction == 'down' and self.blend(position=(posY+1, posX)):
            self.tetromino_position = (posY+1, posX)
            return self.tetromino_position
        else:
            return False


    def rotated(self, rotation=None, tetromino=None):
        if rotation is None:
            rotation = self.tetromino_rotation
        if tetromino == None:
            return rotate(self.current_tetromino.shape, rotation)
        else:
            return rotate(tetromino.shape, rotation)
    
    def block(self, color, shadow=False):
        colors = {'blue':   (27, 34, 224),
                  'yellow': (225, 242, 41),
                  'pink':   (242, 41, 195),
                  'green':  (22, 181, 64),
                  'red':    (204, 22, 22),
                  'orange': (245, 144, 12),
                  'cyan':   (10, 255, 226)}


        if shadow:
            end = [40] # end is the alpha value
        else:
            end = [] # Adding this to the end will not change the array, thus no alpha value

        border = Surface((self.blocksize, self.blocksize), pygame.SRCALPHA, 32)
        border.fill(map(lambda c: c*0.5, colors[color]) + end)

        borderwidth = 2

        box = Surface((self.blocksize-borderwidth*2, self.blocksize-borderwidth*2), pygame.SRCALPHA, 32)
        boxarr = pygame.PixelArray(box)
        for x in range(len(boxarr)):
            for y in range(len(boxarr)):
                boxarr[x][y] = tuple(map(lambda c: min(255, int(c*random.uniform(0.8, 1.2))), colors[color]) + end) 

        del boxarr # deleting boxarr or else the box surface will be 'locked' or something like that and won't blit.
        border.blit(box, Rect(borderwidth, borderwidth, 0, 0))

        return border

    def lock_tetromino(self):

        self.matrix = self.blend()
        if self.matrix is False:
            return False

        lines_cleared = self.remove_lines()
        self.lines += lines_cleared

        if lines_cleared:
            # if lines_cleared >= 4:
            #     self.linescleared_sound.play()
                
            self.score += 100 * (lines_cleared**2) * self.combo
            print "lines cleared!"

            if not self.played_highscorebeaten_sound and self.score > self.highscore:
                # if self.highscore != 0:
                    #self.highscorebeaten_sound.play()
                self.played_highscorebeaten_sound = True

        # if self.lines >= self.level*10:
        #     self.levelup_sound.play()
        #     self.level += 1

        self.combo = self.combo + 1 if lines_cleared else 1

        self.set_tetrominoes()

    def remove_lines(self):
        lines = []
        for y in range(self.size['height']):
            line = (y, [])
            for x in range(self.size['width']):
                if self.matrix[(y,x)]:
                    line[1].append(x)
            if len(line[1]) == self.size['width']:
                lines.append(y)

        for line in sorted(lines):
            for x in range(self.size['width']):
                self.matrix[(line,x)] = None
            for y in range(0, line+1)[::-1]:
                for x in range(self.size['width']):
                    self.matrix[(y,x)] = self.matrix.get((y-1,x), None)

        return len(lines)

    def blend(self, shape=None, position=None, matrix=None, block=None, allow_failure=True, shadow=False):
        """
        Creates a blended matrix with the shadow block, and the current active block.
        
        A blend is ok if the poisition given of the current block does not occupy the space of another block.
        """
        if shape is None:
            shape = self.rotated()
        if position is None:
            position = self.tetromino_position

        copy = dict(self.matrix if matrix is None else matrix)
        posY, posX = position
        for x in range(posX, posX+len(shape)):
            for y in range(posY, posY+len(shape)):
                if (copy.get((y, x), False) is False and shape[y-posY][x-posX] # shape is outside the matrix
                    or # coordinate is occupied by something else which isn't a shadow
                    copy.get((y,x)) and shape[y-posY][x-posX] and copy[(y,x)][0] != 'shadow'): 
                    if allow_failure:
                        return False
                    else:
                        raise BrokenMatrixException("Tried to blend a broken matrix. This should mean game over, if you see this it is certainly a bug. (or you are developing)")
                elif shape[y-posY][x-posX] and not shadow:
                    copy[(y,x)] = ('block', self.tetromino_block if block is None else block)
                    pass
                elif shape[y-posY][x-posX] and shadow:
                    copy[(y,x)] = ('shadow', block)
                    pass
        return copy

    def construct_surface_of_next_tetromino(self):
        shape = self.next_tetromino.shape
        surf = Surface((len(shape)*self.blocksize, len(shape)*self.blocksize), pygame.SRCALPHA, 32)

        for y in range(len(shape)):
            for x in range(len(shape)):
                if shape[y][x]:
                    surf.blit(self.block(self.next_tetromino.color), (x*self.blocksize, y*self.blocksize))
        return surf

class Game(object):
    def __init__(self):
        self.time = 0
    def main(self, screen, callback):
        clock = pygame.time.Clock()
        self.time += 1
        
        background = Surface(screen.get_size())#
        background.blit(construct_nightmare(background.get_size()), (0,0))

        self.matris = Matris()
        matris_border = Surface((MATRIX_WIDTH*BLOCKSIZE+BORDERWIDTH*2, VISIBLE_MATRIX_HEIGHT*BLOCKSIZE+BORDERWIDTH*2))
        matris_border.fill(BORDERCOLOR)
        while 1:
            dt = clock.tick(45)
            
            self.matris.update((dt / 1000.) if not self.matris.paused else 0)
            if self.matris.gameover:
                print "gameover"
                return
                
            callback(self.matris, self.time)

            tricky_centerx = WIDTH-(WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2))/2

            background.blit(matris_border, (MATRIS_OFFSET,MATRIS_OFFSET))
            background.blit(self.matris.surface, (MATRIS_OFFSET+BORDERWIDTH, MATRIS_OFFSET+BORDERWIDTH))

            nextts = self.next_tetromino_surf(self.matris.surface_of_next_tetromino)
            background.blit(nextts, nextts.get_rect(top=MATRIS_OFFSET, centerx=tricky_centerx))

            infos = self.info_surf()
            background.blit(infos, infos.get_rect(bottom=HEIGHT-MATRIS_OFFSET, centerx=tricky_centerx))

            screen.blit(background, (0, 0))

            pygame.display.flip()
    

    def info_surf(self):

        textcolor = (255, 255, 255)
        font = pygame.font.Font(None, 30)
        width = (WIDTH-(MATRIS_OFFSET+BLOCKSIZE*MATRIX_WIDTH+BORDERWIDTH*2)) - MATRIS_OFFSET*2

        def renderpair(text, val):
            text = font.render(text, True, textcolor)
            val = font.render(str(val), True, textcolor)

            surf = Surface((width, text.get_rect().height + BORDERWIDTH*2), pygame.SRCALPHA, 32)

            surf.blit(text, text.get_rect(top=BORDERWIDTH+10, left=BORDERWIDTH+10))
            surf.blit(val, val.get_rect(top=BORDERWIDTH+10, right=width-(BORDERWIDTH+10)))
            return surf

        scoresurf = renderpair("Score", self.matris.score)
        levelsurf = renderpair("Level", self.matris.level)
        linessurf = renderpair("Lines", self.matris.lines)
        combosurf = renderpair("Combo", "x{}".format(self.matris.combo))

        height = 20 + (levelsurf.get_rect().height + 
                       scoresurf.get_rect().height +
                       linessurf.get_rect().height + 
                       combosurf.get_rect().height )

        area = Surface((width, height))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, width-BORDERWIDTH*2, height-BORDERWIDTH*2))

        area.blit(levelsurf, (0,0))
        area.blit(scoresurf, (0, levelsurf.get_rect().height))
        area.blit(linessurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height))
        area.blit(combosurf, (0, levelsurf.get_rect().height + scoresurf.get_rect().height + linessurf.get_rect().height))

        return area

    def next_tetromino_surf(self, tetromino_surf):
        area = Surface((BLOCKSIZE*5, BLOCKSIZE*5))
        area.fill(BORDERCOLOR)
        area.fill(BGCOLOR, Rect(BORDERWIDTH, BORDERWIDTH, BLOCKSIZE*5-BORDERWIDTH*2, BLOCKSIZE*5-BORDERWIDTH*2))

        areasize = area.get_size()[0]
        tetromino_surf_size = tetromino_surf.get_size()[0]
        # ^^ I'm assuming width and height are the same

        center = areasize/2 - tetromino_surf_size/2
        area.blit(tetromino_surf, (center, center))

        return area

class Menu(object):
    running = True

    def main(self, screen, ai_playing=False):
        clock = pygame.time.Clock()
        menu = kezmenu.KezMenu(
            ['Play!', lambda: Game().main(screen)],
            ['Quit', lambda: setattr(self, 'running', False)],
        )
        self.screen = screen
        if ai_playing:
            Game().main(screen)
            
        menu.position = (50, 50)
        menu.enableEffect('enlarge-font-on-focus', font=None, size=60, enlarge_factor=1.2, enlarge_time=0.3)
        menu.color = (255,255,255)
        menu.focus_color = (40, 200, 40)

        nightmare = construct_nightmare(screen.get_size())
        highscoresurf = self.construct_highscoresurf()

        timepassed = clock.tick(30) / 1000.

        while self.running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    exit()

            menu.update(events, timepassed)

            timepassed = clock.tick(30) / 1000.

            if timepassed > 1: # A game has most likely been played 
                highscoresurf = self.construct_highscoresurf()

            screen.blit(nightmare, (0,0))
            screen.blit(highscoresurf, highscoresurf.get_rect(right=WIDTH-50, bottom=HEIGHT-50))
            menu.draw(screen)
            pygame.display.flip()

    def construct_highscoresurf(self):
        font = pygame.font.Font(None, 50)
        highscore = load_score()
        text = "Highscore: {}".format(highscore)
        return font.render(text, True, (255,255,255))


def construct_nightmare(size):
    surf = Surface(size)

    boxsize = 8
    bordersize = 1
    vals = '1235' # only the lower values, for darker colors and greater fear
    arr = pygame.PixelArray(surf)
    for x in xrange(0, len(arr), boxsize):
        for y in xrange(0, len(arr[x]), boxsize):

            color = int(''.join([random.choice(vals) + random.choice(vals) for _ in range(3)]), 16)

            for LX in xrange(x, x+(boxsize - bordersize)):
                for LY in xrange(y, y+(boxsize - bordersize)):
                    if LX < len(arr) and LY < len(arr[x]):
                        arr[LX][LY] = color
    del arr
    return surf

 
def save_object(obj, filename):
    with open(filename, 'ab') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
def open_object(filename):
    with open(filename, 'rb') as input:
        m1 = pickle.load(input)
# save_object(Game(), '/Users/wenqin/Documents/GitHub/grade-12-assignments-wenqinYe/Culminating/memory.pkl')a
#a = open_object('/Users/wenqin/Documents/GitHub/grade-12-assignments-wenqinYe/Culminating/memory.pkl')

def callback(a, b):
    pass
    
    
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MaTris")
    game = Game()
    while(1):
        game.main(screen, callback)