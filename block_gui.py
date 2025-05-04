# -----------------------------------------------
# filename: block_gui.py
# Changes for Phase 2: Visual Ports
# -----------------------------------------------
"""
Dynamic BlockSim editor - Phase 2: Visual Ports
  - Shows named input/output ports on blocks based on sim_blocks.py definitions.
  - Connections still manual via generic block clicks (Phase 3 needed for port-to-port).
"""
import sys
import os
import json
import re
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
import importlib.util
from redbaron import RedBaron
from collections import defaultdict

# --- Import SimBlock classes ---
# Add error handling in case sim_blocks.py is missing or has issues
try:
    # Ensure the directory containing sim_blocks.py is in the path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from sim_blocks import SimBlock, Constant, Sum, Integrator, Plot
    SIM_BLOCKS_LOADED = True

    # --- Map block type names (strings) to actual classes ---
    BLOCK_TYPE_MAP = {
        'Constant': Constant,
        'Sum': Sum,
        'Integrator': Integrator,
        'Plot': Plot,
        # Add other block types here as they are created
        # 'Signal Generator': SignalGenerator, # Assuming SignalGenerator exists
    }
except ImportError as e:
    print(f"ERROR: Could not import from sim_blocks.py: {e}")
    print("Ensure sim_blocks.py is in the same directory or Python path.")
    messagebox.showerror("Import Error", f"Failed to load simulation blocks: {e}\nGUI functionality will be limited.")
    SIM_BLOCKS_LOADED = False
    BLOCK_TYPE_MAP = {} # Empty map if import failed
except Exception as e:
    print(f"ERROR: Unexpected error during import: {e}")
    messagebox.showerror("Import Error", f"Unexpected error loading simulation blocks: {e}\nGUI functionality will be limited.")
    SIM_BLOCKS_LOADED = False
    BLOCK_TYPE_MAP = {}


# Global reference for callbacks
current_app = None

# --- Block Class ---
class Block:
    PORT_R = 5 # Radius for port shapes
    PORT_SPACING_MIN = 15 # Minimum pixels between ports
    PORT_LABEL_OFFSET = 3 # Pixels between port shape and label
    INPUT_PORT_COLOR = "#4a90e2" # Blueish
    OUTPUT_PORT_COLOR = "#50e3c2" # Greenish

    def __init__(self, canvas, bid, x, y, w, h, title, code, ports_info):
        self.canvas = canvas
        self.bid = bid
        self.code_text = code
        self.ports_info = ports_info if ports_info else {'inputs': {}, 'outputs': {}} # Store port metadata
        self.title = title

        # --- Main Block Rectangle and Text ---
        self.id = canvas.create_rectangle(x, y, x+w, y+h,
                                          fill="#ddd", outline="#555", width=2, tags=f"block_{bid}")
        self.text = canvas.create_text(x+5, y+5,
                                       anchor="nw", text=title,
                                       font=("Arial", 10, "bold"), tags=f"block_{bid}")
        display_code = (code[:30] + '...') if len(code) > 30 else code
        self.code = canvas.create_text(x+5, y+h-5,
                                       anchor="sw", text=display_code,
                                       font=("Arial", 8), tags=f"block_{bid}")

        # --- Create Named Ports (Inputs on Left, Outputs on Right) ---
        self.input_ports = {} # port_name -> {'id': canvas_id, 'label_id': label_id, 'x': cx, 'y': cy}
        self.output_ports = {} # Same structure

        # Draw Input Ports
        input_names = list(self.ports_info.get('inputs', {}).keys())
        num_inputs = len(input_names)
        input_spacing = max(self.PORT_SPACING_MIN, h / (num_inputs + 1)) if num_inputs > 0 else h / 2
        for i, port_name in enumerate(input_names):
            port_x = x
            port_y = y + (i + 1) * input_spacing
            port_id = canvas.create_oval(port_x - self.PORT_R, port_y - self.PORT_R,
                                         port_x + self.PORT_R, port_y + self.PORT_R,
                                         fill=self.INPUT_PORT_COLOR, outline="#333", tags=(f"block_{bid}", "port", "input_port"))
            label_x = port_x + self.PORT_R + self.PORT_LABEL_OFFSET
            label_id = canvas.create_text(label_x, port_y, text=port_name, anchor="w",
                                          font=("Arial", 7), tags=(f"block_{bid}", "port_label"))
            self.input_ports[port_name] = {'id': port_id, 'label_id': label_id, 'x': port_x, 'y': port_y}

        # Draw Output Ports
        output_names = list(self.ports_info.get('outputs', {}).keys())
        num_outputs = len(output_names)
        output_spacing = max(self.PORT_SPACING_MIN, h / (num_outputs + 1)) if num_outputs > 0 else h / 2
        for i, port_name in enumerate(output_names):
            port_x = x + w
            port_y = y + (i + 1) * output_spacing
            port_id = canvas.create_oval(port_x - self.PORT_R, port_y - self.PORT_R,
                                         port_x + self.PORT_R, port_y + self.PORT_R,
                                         fill=self.OUTPUT_PORT_COLOR, outline="#333", tags=(f"block_{bid}", "port", "output_port"))
            label_x = port_x - self.PORT_R - self.PORT_LABEL_OFFSET
            label_id = canvas.create_text(label_x, port_y, text=port_name, anchor="e",
                                          font=("Arial", 7), tags=(f"block_{bid}", "port_label"))
            self.output_ports[port_name] = {'id': port_id, 'label_id': label_id, 'x': port_x, 'y': port_y}


        # --- Bind Events ---
        # Bind drag/hover/delete to main rectangle and text labels
        for item_tag in (self.id, self.text, self.code):
             canvas.tag_bind(item_tag, '<ButtonPress-1>', self._on_press)
             canvas.tag_bind(item_tag, '<B1-Motion>', self._on_drag)
             canvas.tag_bind(item_tag, '<Enter>', self._on_enter)
             canvas.tag_bind(item_tag, '<Leave>', self._on_leave)
             canvas.tag_bind(item_tag, '<Button-3>', self._on_right_click)

        # Bind code edit
        canvas.tag_bind(self.code, '<Double-1>', self._on_double)

        # --- Bind Port Clicks (for Phase 3) ---
        # Store port name in the binding callback
        for port_name, port_data in self.input_ports.items():
             canvas.tag_bind(port_data['id'], '<Button-1>',
                             lambda event, name=port_name: self._on_in_port_click(event, name))
        for port_name, port_data in self.output_ports.items():
             canvas.tag_bind(port_data['id'], '<Button-1>',
                             lambda event, name=port_name: self._on_out_port_click(event, name))


    def bbox(self):
        if self.canvas.winfo_exists() and self.id in self.canvas.find_all():
            return self.canvas.coords(self.id)
        return [0, 0, 0, 0]

    def get_port_coord(self, port_name):
        """Returns the (x, y) coordinates of the center of a named port."""
        if port_name in self.input_ports:
            data = self.input_ports[port_name]
            return (data['x'], data['y'])
        elif port_name in self.output_ports:
            data = self.output_ports[port_name]
            return (data['x'], data['y'])
        return None # Port not found


    def _on_press(self, evt):
        # Raise block components (including ports and labels) above others
        self.canvas.tag_raise(f"block_{self.bid}")
        x1, y1, *_ = self.bbox()
        # Make sure coords are valid numbers
        if isinstance(x1, (int, float)) and isinstance(y1, (int, float)):
            self._dx = evt.x - x1
            self._dy = evt.y - y1
        else: # Fallback if coords are weird
             self._dx = evt.x
             self._dy = evt.y


    def _on_drag(self, evt):
        # Clamp coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        x1, y1, x2, y2 = self.bbox()
        w = x2 - x1
        h = y2 - y1
        if not all(isinstance(c, (int, float)) for c in [x1, y1, w, h]) or w <= 0 or h <= 0:
             # print(f"Warning: Invalid bbox for block {self.bid} on drag start: {[x1, y1, x2, y2]}")
             return # Avoid calculations with invalid dimensions

        nx = max(0, min(evt.x - self._dx, canvas_width - w))
        ny = max(0, min(evt.y - self._dy, canvas_height - h))

        move_dx, move_dy = nx - x1, ny - y1

        if abs(move_dx) > 0 or abs(move_dy) > 0:
            # Move all parts of the block using the tag
            self.canvas.move(f"block_{self.bid}", move_dx, move_dy)

            # --- Update stored port coordinates ---
            for port_dict in (self.input_ports, self.output_ports):
                 for port_name, port_data in port_dict.items():
                      port_data['x'] += move_dx
                      port_data['y'] += move_dy

            # --- Update connected lines (Needs Connection class update) ---
            conns_to_update = list(Connection._instances)
            for conn in conns_to_update:
                 if conn.src == self or conn.dst == self:
                    conn.update() # Connection.update needs to use get_port_coord


    def _on_enter(self, _):
        if self.canvas.winfo_exists() and self.id in self.canvas.find_all():
            self.canvas.itemconfigure(self.id, width=4, outline='#333')

    def _on_leave(self, _):
        if self.canvas.winfo_exists() and self.id in self.canvas.find_all():
            self.canvas.itemconfigure(self.id, width=2, outline='#555')

    # --- Port Click Handlers (Phase 3) ---
    def _on_out_port_click(self, event, port_name):
        """Called when an output port shape is clicked."""
        print(f"Clicked output port: {port_name} on block {self.bid}")
        current_app.start_connection(self, port_name)
        return 'break'

    def _on_in_port_click(self, event, port_name):
        """Called when an input port shape is clicked."""
        print(f"Clicked input port: {port_name} on block {self.bid}")
        current_app.finish_connection(self, port_name)
        return 'break'

    # --- Other Handlers ---
    def _on_double(self, _):
        # (Code from previous version - remains the same for now)
        """Edit the block's code."""
        new_code = simpledialog.askstring('Edit Code', f'Edit code for Block {self.bid}:',
                                          initialvalue=self.code_text)
        if new_code is not None and new_code != self.code_text:
            self.code_text = new_code.strip() # Store the edited code
            # Update visual text (ellipsized)
            display_code = (self.code_text[:30] + '...') if len(self.code_text) > 30 else self.code_text
            self.canvas.itemconfigure(self.code, text=display_code)

            # Update the RedBaron AST immediately
            if not current_app or not current_app.red:
                 messagebox.showwarning("Code Edit", "Cannot update code in source file: Model AST not loaded.")
                 return

            func = current_app.red.find('def', name='my_model')
            if not func:
                 messagebox.showwarning("Code Edit", "Cannot update code in source file: 'my_model' function not found.")
                 return

            code_node_index = -1
            for i, node in enumerate(func.value):
                 if node.type == 'comment' and f'blockid={self.bid}' in node.value:
                     if i + 1 < len(func.value) and func.value[i+1].type != 'comment':
                         code_node_index = i + 1
                         break

            if code_node_index != -1:
                try:
                    new_nodes = RedBaron(self.code_text + '\n')
                    if new_nodes:
                         func.value[code_node_index] = new_nodes[0]
                    else:
                         del func.value[code_node_index]
                         messagebox.showwarning("Code Edit", f"Code for block {self.bid} was cleared. Corresponding comment remains.")

                except Exception as e:
                    messagebox.showerror("Code Error", f"Failed to parse new code: {e}\nCode change not applied to AST.")
            else:
                 messagebox.showwarning("Code Edit", f"Could not find the code line associated with block {self.bid} in the source file to update.")


    def _on_right_click(self, event):
        """Callback for right-clicking the block."""
        current_app.delete_block(self)
        return "break"

    def destroy_visuals(self):
        """Safely delete all canvas items associated with this block."""
        if not self.canvas.winfo_exists(): return
        try:
             self.canvas.delete(f"block_{self.bid}") # Delete all items tagged for this block
        except tk.TclError as e:
             print(f"Warning: Minor error deleting canvas items for block {self.bid}: {e}")
        finally:
             # Clear references, just in case
             self.input_ports.clear()
             self.output_ports.clear()
             self.id = None
             self.text = None
             self.code = None



# --- Connection Class (Needs Update for Port Coords) ---
class Connection:
    _instances = []

    def __init__(self, canvas, src_block: Block, src_port: str, dst_block: Block, dst_port: str):
        self.canvas = canvas
        self.src = src_block
        self.src_port = src_port
        self.dst = dst_block
        self.dst_port = dst_port

        self.line = canvas.create_line(0,0,0,0, # Initial dummy coords
                                       width=2, smooth=True,
                                       arrow='last', arrowshape=(8,10,3),
                                       tags="connection_line")
        canvas.tag_bind(self.line, "<Button-3>", self._on_right_click)
        # Lower the line so it's drawn below blocks
        canvas.tag_lower(self.line)

        Connection._instances.append(self)
        self.update() # Calculate initial coords

    def update(self):
        """Updates line coordinates based on connected port positions."""
        if not all([self.src, self.dst, self.canvas.winfo_exists(),
                    self.line in self.canvas.find_all()]):
             # If block or canvas gone, schedule self for destruction
             if self in Connection._instances:
                  # Use after to avoid issues during canvas updates
                  self.canvas.after_idle(self.destroy)
             return

        # Get coordinates from the specific ports using the helper method
        src_coords = self.src.get_port_coord(self.src_port)
        dst_coords = self.dst.get_port_coord(self.dst_port)

        if src_coords and dst_coords:
            ox, oy = src_coords
            ix, iy = dst_coords
            # Simple bezier-like curve calculation
            mx = (ix + ox) / 2
            # Draw line using waypoints for the curve
            self.canvas.coords(self.line, ox, oy, mx, oy, mx, iy, ix, iy)
        else:
            # Hide line if ports not found (e.g., block structure changed)
            self.canvas.coords(self.line, 0, 0, 0, 0)
            print(f"Warning: Could not find ports for connection {self.src.bid}:{self.src_port} -> {self.dst.bid}:{self.dst_port}")


    def _on_right_click(self, event):
        current_app.delete_connection(self)
        return "break"

    def destroy(self):
        """Removes the connection visually and from the global list."""
        try:
             if self.canvas.winfo_exists() and self.line in self.canvas.find_all():
                  self.canvas.delete(self.line)
        except tk.TclError:
             pass
        finally:
             if self in Connection._instances:
                  Connection._instances.remove(self)
             self.src = None
             self.dst = None
             self.src_port = None
             self.dst_port = None


# --- Utility Functions (parse_model, infer_connections, title_from_code) ---
# These remain largely the same for now, but title_from_code might need adjustment later
# if code snippets become less reliable for determining type.

def parse_model(path):
    try:
        with open(path, 'r') as f: source_code = f.read()
        red = RedBaron(source_code)
    except FileNotFoundError:
         messagebox.showerror("Load Error", f"File not found: {path}")
         return None, []
    except Exception as e:
        messagebox.showerror("Parse Error", f"Failed to parse {path} with RedBaron:\n{e}")
        return None, []

    func = red.find('def', name='my_model')
    if not func:
        # Check if code is empty or just imports/comments
        if len(red) < 3: # Arbitrary small number, adjust as needed
            print("Model file seems empty or lacks 'my_model' function. Proceeding with empty canvas.")
            return red, []
        messagebox.showerror("Parse Error", f"Could not find 'def my_model(t, dt):' in {path}")
        return red, []

    blocks = []
    nodes_to_process = list(func.value)
    i = 0
    while i < len(nodes_to_process):
        node = nodes_to_process[i]
        if node.type == 'comment' and 'blockid=' in node.value:
            meta = {}
            try:
                parts = node.value.lstrip('# ').split(';')
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        meta[key.strip()] = value.strip()

                bid = int(meta.get('blockid', -1))
                if bid == -1: raise ValueError("blockid missing")
                x = int(meta.get('x', 50))
                y = int(meta.get('y', 50))
                # Connections string ('con=') is parsed but not used yet for block creation itself

                code_line_node = None
                if i + 1 < len(nodes_to_process):
                     potential_code_node = nodes_to_process[i+1]
                     if potential_code_node.type not in ('comment', 'pass'):
                         code_line_node = potential_code_node
                         i += 1 # Skip code node next iteration

                if code_line_node:
                    blocks.append({'id': bid, 'x': x, 'y': y,
                                   'code': code_line_node.dumps().strip()})
                else:
                    # Tolerate missing code line for now, block might still be drawn
                    print(f"Warning: Found block comment for id={bid} but no valid code line followed.")
                    # Add a placeholder block structure if code is missing?
                    blocks.append({'id': bid, 'x': x, 'y': y, 'code': f'# Missing code for block {bid}'})


            except (ValueError, KeyError, TypeError) as e:
                print(f"Warning: Could not parse block metadata or find code: {node.value.strip()}. Error: {e}")
            except Exception as e:
                print(f"Warning: Unexpected error processing block comment: {node.value.strip()}. Error: {e}")
        i += 1
    return red, blocks


def title_from_code(code):
    # Attempt to guess type from code for display title
    # This becomes less reliable; better to store type explicitly if possible
    if 'Constant(' in code or re.match(r'\w+\s*=\s*[\d\.]+$', code): return 'Constant'
    if 'Integrator(' in code or '.update' in code: return 'Integrator'
    if 'Sum(' in code or re.search(r'=\s*\w+\s*\+\s*\w+', code): return 'Sum'
    if 'Plot(' in code or 'scope.log' in code: return 'Plot' # Adapt if scope still used
    # Add more specific checks based on sim_blocks.py contents
    return 'Custom' # Fallback

# infer_connections remains the same - it's a fallback if JSON is missing/bad

# --- BlockSimApp Class ---
class BlockSimApp:
    def __init__(self, root, canvas):
        global current_app; current_app=self
        self.root = root; self.cv = canvas
        self.current_path=None; self.red=None; self.objs={} # maps bid -> Block object
        # Store pending connection source block AND port name
        self.pending_connection = None # tuple: (src_block, src_port_name)
        self._make_menu(); self._make_ctx()
        self.cv.bind('<Button-3>', self._show_ctx)

    def _make_menu(self):
        # (No changes needed here)
        mb=tk.Menu(self.root); fm=tk.Menu(mb, tearoff=0)
        fm.add_command(label='Load...',command=self.on_load)
        fm.add_command(label='Save', command=self.on_save)
        fm.add_command(label='Run',  command=self.on_run)
        mb.add_cascade(label='File',menu=fm); self.root.config(menu=mb)

    def _make_ctx(self):
        # (Populate from BLOCK_TYPE_MAP keys)
        self.ctx=tk.Menu(self.root,tearoff=0)
        if not SIM_BLOCKS_LOADED:
             self.ctx.add_command(label="Cannot insert blocks (Import Error)", state="disabled")
             return

        # Use the keys from BLOCK_TYPE_MAP
        for typ in BLOCK_TYPE_MAP.keys():
            self.ctx.add_command(label=f"Insert {typ}",
                                 command=lambda t=typ: self.insert_block(t))

    def _show_ctx(self,e):
        # (Logic to check if clicking on empty space remains the same)
        overlapping = self.cv.find_overlapping(e.x-1, e.y-1, e.x+1, e.y+1)
        is_on_item = False
        for item_id in overlapping:
            tags = self.cv.gettags(item_id)
            if "connection_line" in tags or "block_" in "".join(tags): # Check tags
                 is_on_item = True
                 break
        if not is_on_item:
            self.menu_x, self.menu_y = e.x,e.y
            self.ctx.tk_popup(e.x_root,e.y_root)


    # --- Connection Handling (Updated for Ports) ---
    def start_connection(self, src_block: Block, src_port_name: str):
        """Initiates drawing a connection line from a specific port."""
        self.pending_connection = (src_block, src_port_name)
        self.cv.config(cursor='crosshair')
        print(f"Starting connection from {src_block.bid}:{src_port_name}")

    def finish_connection(self, dst_block: Block, dst_port_name: str):
        """Completes a visual connection if one is pending."""
        if self.pending_connection:
            src_block, src_port_name = self.pending_connection
            # Don't connect output to output or input to input (basic check)
            if src_port_name in src_block.output_ports and dst_port_name in dst_block.input_ports:
                 # Check if connection already exists visually
                 exists = any(conn.src == src_block and conn.src_port == src_port_name and
                              conn.dst == dst_block and conn.dst_port == dst_port_name
                              for conn in Connection._instances)
                 if not exists:
                      try:
                          Connection(self.cv, src_block, src_port_name, dst_block, dst_port_name)
                          print(f"Visual connection created: {src_block.bid}:{src_port_name} -> {dst_block.bid}:{dst_port_name}")
                          # Remove the old messagebox, rely on visual feedback
                      except Exception as e:
                           print(f"Error creating Connection object: {e}")
                           messagebox.showerror("Connection Error", f"Failed to create connection: {e}")
                 else:
                      print(f"Connection {src_block.bid}:{src_port_name} -> {dst_block.bid}:{dst_port_name} already exists.")
            else:
                 print("Connection Error: Cannot connect output-to-output or input-to-input.")
        else:
             print("Finish connection called but no connection pending.")

        self.pending_connection = None
        self.cv.config(cursor='')

    # --- Block Deletion ---
    def delete_block(self, block_to_delete: Block):
        # (Needs to also delete port visuals - handled by block.destroy_visuals)
        if not self.red:
             # Allow deleting GUI elements even if AST isn't loaded? Yes.
             print("Warning: Model AST not loaded. Deleting block from GUI only.")
        elif not messagebox.askyesno("Confirm Delete", f"Delete block {block_to_delete.bid} ('{block_to_delete.title}') and its connections?", parent=self.root):
            return

        bid = block_to_delete.bid

        # --- Delete Connections ---
        conns_to_remove = [conn for conn in Connection._instances if conn.src == block_to_delete or conn.dst == block_to_delete]
        for conn in conns_to_remove:
            conn.destroy()

        # --- Delete GUI Elements ---
        block_to_delete.destroy_visuals()

        # --- Remove from internal tracking ---
        if bid in self.objs:
            del self.objs[bid]
        else:
             print(f"Warning: Block {bid} not found in internal objects dictionary during deletion.")

        # --- Remove from RedBaron AST (only if loaded) ---
        if self.red:
            func = self.red.find('def', name='my_model')
            if func:
                 comment_index, code_index = -1, -1
                 for i, node in enumerate(func.value):
                     if node.type == 'comment' and f'blockid={bid}' in node.value:
                         comment_index = i
                         if i + 1 < len(func.value) and func.value[i+1].type != 'comment':
                             code_index = i + 1
                         break
                 # Delete nodes carefully
                 if code_index != -1: del func.value[code_index]
                 if comment_index != -1:
                      # Adjust index if code was before it (shouldn't happen) or deleted
                      if code_index != -1 and code_index < comment_index: comment_index -= 1
                      # Handle case where comment_index might be invalid if code was deleted
                      if comment_index >= 0 and comment_index < len(func.value):
                            del func.value[comment_index]
                      else:
                          print(f"Warning: Comment index {comment_index} became invalid after code deletion for block {bid}.")
                 print(f"Removed block {bid} from AST (if found).")
            else:
                print("Warning: 'my_model' function not found in AST during block deletion.")


    def delete_connection(self, conn_to_delete: Connection):
         # (Message needs update for ports)
         src_bid = conn_to_delete.src.bid if conn_to_delete.src else '?'
         src_port = conn_to_delete.src_port if conn_to_delete.src_port else '?'
         dst_bid = conn_to_delete.dst.bid if conn_to_delete.dst else '?'
         dst_port = conn_to_delete.dst_port if conn_to_delete.dst_port else '?'

         if messagebox.askyesno("Confirm Delete", f"Delete connection\nFrom: Block {src_bid} Port '{src_port}'\nTo:     Block {dst_bid} Port '{dst_port}'?", parent=self.root):
            conn_to_delete.destroy()
            print(f"Deleted connection: {src_bid}:{src_port} -> {dst_bid}:{dst_port}")

    # --- Load / Save / Run ---
    def on_load(self):
        # (Needs update to pass ports_info to Block constructor)
        p=filedialog.askopenfilename(defaultextension=".py",
                                     filetypes=[('Python Files','*.py'), ('All Files','*.*')])
        if p:
             self.cv.delete('all')
             Connection._instances.clear()
             self.objs.clear()
             self.red = None
             self.current_path = None
             self.pending_connection = None
             self.cv.config(cursor='')
             self.load_model(p)

    def on_save(self):
        # (Needs update to save port connections to JSON)
        if not self.current_path or not self.red:
            messagebox.showwarning('Save','No model loaded or AST missing.', parent=self.root); return

        func=self.red.find('def',name='my_model')
        if not func:
             messagebox.showerror('Save Error', "Could not find 'my_model' function in the AST.", parent=self.root)
             return

        # Update comments (position only for now, connections go to JSON)
        incoming_connections_comment = defaultdict(list) # Keep for potential future use in comments
        for conn in Connection._instances:
             if conn.src and conn.dst:
                 incoming_connections_comment[conn.dst.bid].append(conn.src.bid)

        nodes_to_process = list(func.value)
        found_ids_in_code = set()
        for i, node in enumerate(nodes_to_process):
            if node.type=='comment' and 'blockid=' in node.value:
                # ... (parsing meta as before) ...
                try:
                    meta = {}
                    parts = node.value.lstrip('# ').split(';')
                    for part in parts:
                        if '=' in part: key, value = part.split('=', 1); meta[key.strip()] = value.strip()
                    bid = int(meta.get('blockid', -1))
                    if bid == -1: raise ValueError("blockid missing")
                    found_ids_in_code.add(bid)

                    if bid in self.objs:
                        blk = self.objs[bid]
                        x, y, *_ = blk.bbox()
                        # Keep comment simple for now, just position
                        func.value[i].value = f'# blockid={bid};x={int(x)};y={int(y)}'
                    # else: block in code but not GUI - already warned during load/delete
                except Exception as e:
                     print(f"Warning: Error updating comment node {node.value}. Error: {e}")


        # Write Python file
        try:
            with open(self.current_path,'w') as f: f.write(self.red.dumps())
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to write Python file:\n{e}", parent=self.root); return

        # --- Write DETAILED connectors JSON ---
        conns_data = []
        for c in Connection._instances:
             if c.src and c.dst and c.src_port and c.dst_port:
                 conns_data.append([c.src.bid, c.src_port, c.dst.bid, c.dst_port])
             else:
                 print(f"Warning: Skipping save to JSON for invalid connection data: {c}")

        try:
            jpath = self.current_path + '.connections.json'
            with open(jpath, 'w') as jf:
                json.dump(conns_data, jf, indent=2)
        except Exception as e:
             messagebox.showwarning("Save Warning", f"Failed to write connections JSON file:\n{e}\nConnections might not be fully saved.", parent=self.root)

        messagebox.showinfo('Save','Saved model code and connections JSON.', parent=self.root)


    def on_run(self):
        # --- This needs major update for Phase 5 (Simulation Engine) ---
        messagebox.showinfo("Run", "Simulation execution needs to be updated for the new Port-based system (Phase 5).\nRunning the old code directly will likely fail or ignore visual connections.", parent=self.root)
        # (Keep old run logic commented out or behind a check for now)
        return

        # OLD RUN LOGIC (for reference, but likely won't work correctly)
        # if not self.current_path: messagebox.showwarning('Run', ...); return
        # save_choice = messagebox.askyesnocancel(...)
        # ...
        # try:
        #     ... (importlib stuff) ...
        #     m.simulate(m.my_model, ...) # This simulate needs changing
        #     m.scope.plot(...) # Plotting needs changing (use Plot block's on_simulation_end)
        # ... (exception handling) ...

    def insert_block(self,typ):
        # (Needs update to fetch and pass ports_info)
        if not self.red or not SIM_BLOCKS_LOADED:
            messagebox.showwarning("Insert Block", "Cannot insert: Load model or check sim_blocks import.", parent=self.root)
            return

        nid=max(list(self.objs.keys()) + [0])+1
        x,y=self.menu_x, self.menu_y

        # Get the class and default code
        block_class = BLOCK_TYPE_MAP.get(typ)
        if not block_class:
             messagebox.showerror("Insert Error", f"Unknown block type: {typ}", parent=self.root)
             return

        # --- Fetch port info ---
        ports_info = {}
        try:
             # Instantiate temporarily to get metadata
             temp_block_instance = block_class()
             ports_info = temp_block_instance.get_ports()
             # Get parameters to potentially create better default code? (Optional)
             # params = temp_block_instance.get_parameters()
        except Exception as e:
             print(f"Error getting metadata for {typ}: {e}")
             messagebox.showwarning("Insert Warning", f"Could not get port info for {typ}. Block may not display correctly.", parent=self.root)


        # Create default code snippet (less critical now, user edits anyway)
        default_code_snippets = { # Simple examples
             'Constant': f'c{nid} = Constant(value=1.0)', # Assumes instance name matches var
             'Sum': f's{nid} = Sum()',
             'Integrator': f'i{nid} = Integrator(initial=0.0)',
             'Plot': f'p{nid} = Plot()'
        }
        sn = default_code_snippets.get(typ, f'# Code for {typ} block {nid}')


        # Create GUI block, passing ports_info
        try:
            blk=Block(self.cv, nid, x, y, 120, 60, typ, sn, ports_info)
            self.objs[nid]=blk
        except Exception as e:
             messagebox.showerror("Insert Error", f"Failed to create GUI block {nid}: {e}", parent=self.root)
             return

        # Add to RedBaron AST (comment only for position)
        func = self.red.find('def', name='my_model')
        if not func:
             messagebox.showerror("Error", "Cannot find 'my_model' function to insert block into.", parent=self.root)
             self.delete_block(blk) # Rollback GUI
             return

        try:
            comment_node = RedBaron(f"# blockid={nid};x={int(x)};y={int(y)}\n")[0]
            code_node = RedBaron(sn + "\n")[0] # Insert basic code line
        except Exception as e:
             messagebox.showerror("Insert Error", f"Failed to create RedBaron nodes: {e}", parent=self.root)
             self.delete_block(blk); return

        insert_index = len(func.value)
        for i, node in enumerate(func.value):
            if node.type == 'return': insert_index = i; break
        try:
             func.value.insert(insert_index, comment_node)
             func.value.insert(insert_index + 1, code_node)
             print(f"Inserted block {nid} ('{typ}') into AST.")
        except Exception as e:
             messagebox.showerror("Insert Error", f"Failed to insert nodes into AST: {e}", parent=self.root)
             self.delete_block(blk)

    def load_model(self,path):
        # (Previous loading parts: clear canvas, parse_model, create Block objects...)
        print(f"Loading model from: {path}")
        self.current_path=path;
        self.cv.delete('all'); Connection._instances.clear(); self.objs.clear()

        self.red, blks_data = parse_model(path)
        if self.red is None: self.current_path = None; return

        # --- Create Block objects (includes getting ports_info) ---
        temp_objs = {}
        for b_data in blks_data:
            bid = b_data['id']
            if bid in temp_objs: continue

            code = b_data.get('code', '# Missing code')
            block_title = title_from_code(code) # Guess title
            block_class = BLOCK_TYPE_MAP.get(block_title)
            ports_info = {}
            if block_class and SIM_BLOCKS_LOADED:
                 try:
                      temp_instance = block_class()
                      ports_info = temp_instance.get_ports()
                 except Exception as e: print(f"Warning: Could not get port info for block {bid} (type {block_title}): {e}")
            elif SIM_BLOCKS_LOADED: print(f"Warning: Could not map title '{block_title}' to class for block {bid}.")

            try:
                blk = Block(self.cv, bid, b_data['x'], b_data['y'], 120, 60,
                            block_title, code, ports_info)
                temp_objs[bid]=blk
            except Exception as e:
                 messagebox.showerror("Load Error", f"Failed to create block object for ID {bid}:\n{e}", parent=self.root)
                 self.cv.delete('all'); self.objs.clear(); self.red=None; self.current_path=None; return

        self.objs = temp_objs
        print(f"Parsed and created {len(self.objs)} blocks.")

        # --- Load connections (from JSON - Preferred Method) ---
        jpath = path + '.connections.json'
        conns_data = [] # Expect list of [src_id, src_port, dst_id, dst_port]
        connections_loaded_from_json = False # Flag to track source

        if os.path.exists(jpath):
            try:
                 print(f"Loading connections from: {jpath}")
                 with open(jpath, 'r') as jf: loaded_data = json.load(jf)

                 # --- Validate JSON format ---
                 if isinstance(loaded_data, list) and all(
                     isinstance(item, (list, tuple)) and len(item) == 4 and
                     isinstance(item[0], int) and isinstance(item[1], str) and # Check types
                     isinstance(item[2], int) and isinstance(item[3], str)
                     for item in loaded_data
                 ):
                     conns_data = loaded_data
                     connections_loaded_from_json = True
                     print(f"Loaded {len(conns_data)} connections from JSON.")
                 elif loaded_data: # Handle non-empty but invalid format
                     print(f"Warning: Invalid format in {jpath}. Expected list of [int:src_id, str:src_port, int:dst_id, str:dst_port].")
                 else:
                     print(f"Connections file {jpath} is empty.")

            except json.JSONDecodeError as e:
                 print(f"Warning: Could not decode JSON from {jpath}: {e}.")
            except Exception as e:
                 print(f"Warning: Error reading {jpath}: {e}.")
        else:
             print(f"Connections file {jpath} not found.")

        # --- Fallback to Inference (Optional - Can be removed if JSON is required) ---
        if not connections_loaded_from_json:
             print("Attempting to infer connections from code (fallback)...")
             inferred_conns = infer_connections(blks_data) # Returns [(src_id, dst_id)]
             # Convert inferred connections to the detailed format with placeholder port names?
             # This is difficult and ambiguous without knowing which ports map to which variables.
             # For now, let's *not* create visual connections based on inference if JSON fails.
             # We only rely on JSON for explicit port-to-port connections.
             print(f"Inferred {len(inferred_conns)} potential block-to-block links (ports unknown).")
             print("INFO: Visual connections are only created from the .connections.json file.")
             conns_data = [] # Ensure we don't use inferred data for visual port connections


        # --- Create Connection objects from loaded JSON data ---
        created_conns_count = 0
        if connections_loaded_from_json:
            for s_id, s_port, d_id, d_port in conns_data:
                # --- Validate connection against loaded blocks and ports ---
                if s_id in self.objs and d_id in self.objs:
                     src_block = self.objs[s_id]
                     dst_block = self.objs[d_id]

                     # Check if the specified ports actually exist on these blocks
                     if s_port in src_block.output_ports and d_port in dst_block.input_ports:
                          try:
                               # Create the visual connection using all details
                               Connection(self.cv, src_block, s_port, dst_block, d_port)
                               created_conns_count += 1
                          except Exception as e:
                               print(f"Error creating visual connection object {s_id}:{s_port}->{d_id}:{d_port}: {e}")
                     else:
                          print(f"Warning: Skipping connection ({s_id}:{s_port} -> {d_id}:{d_port}) - Port name mismatch or port doesn't exist on block '{src_block.title}' or '{dst_block.title}'.")
                          print(f"  > Src Block {s_id} Outputs: {list(src_block.output_ports.keys())}")
                          print(f"  > Dst Block {d_id} Inputs: {list(dst_block.input_ports.keys())}")
                else:
                     # Block ID mentioned in JSON wasn't found/loaded
                     missing_blocks = []
                     if s_id not in self.objs: missing_blocks.append(str(s_id))
                     if d_id not in self.objs: missing_blocks.append(str(d_id))
                     print(f"Warning: Skipping connection ({s_id}:{s_port} -> {d_id}:{d_port}) - Block ID(s) [{', '.join(missing_blocks)}] not found.")

        print(f"Created {created_conns_count} visual connections from JSON data.")

# --- Main Execution ---
def main():
    # (No changes needed here)
    root=tk.Tk();root.title('BlockSim Editor - Phase 2')
    root.geometry("900x700")
    cv=tk.Canvas(root,bg='#f7f7f7',width=800,height=600);cv.pack(fill='both',expand=True)
    app=BlockSimApp(root,cv)
    initial_model = 'my_model.py'
    if len(sys.argv) > 1:
        if os.path.isfile(sys.argv[1]) and sys.argv[1].endswith('.py'): initial_model = sys.argv[1]
        else: print(f"Warning: Arg '{sys.argv[1]}' not valid Python file. Using default.")

    if os.path.exists(initial_model):
        try: app.load_model(initial_model)
        except Exception as e: messagebox.showerror('Initial Load Error',f"Failed to load '{initial_model}':\n{e}", parent=root)
    else:
         messagebox.showinfo("Welcome", f"Model '{initial_model}' not found.\nUse File > Load or create blocks.", parent=root)
         app.red = RedBaron("# BlockSim Model\n\ndef my_model(t, dt):\n    pass\n")
         app.current_path = initial_model

    root.bind('<Escape>',lambda e:root.destroy()); root.mainloop()

if __name__=='__main__':
    main()