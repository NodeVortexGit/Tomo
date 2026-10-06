from pathlib import Path

from tomo.apps import (DesktopApp, SystemCatalog, checked_launch, clean_exec, clean_name, edit_distance, exe_app,
                       installed_apps, main_program, merged, parse_entry, parse_start_apps, pe_subsystem,
                       program_files_apps, registered_apps, resembles, shortcut_target, start_menu_apps, windowed)
from tomo.commands import hard_denied


def program(path: Path, has_windows: bool = True) -> Path:
    """A stand-in Windows program: just the headers that say what kind it is."""
    head = bytearray(256)
    head[:2] = b"MZ"
    head[60:64] = (128).to_bytes(4, "little")
    head[128:132] = b"PE\0\0"
    head[128 + 92:128 + 94] = (2 if has_windows else 3).to_bytes(2, "little")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(head))
    return path


def shortcut(path: Path, target: str) -> Path:
    """A minimal .lnk whose link info names ``target``."""
    volume = (16).to_bytes(4, "little") + bytes(12)
    base_at = 0x1C + len(volume)
    body = volume + target.encode("ascii") + b"\0\0"
    fields = (0x1C + len(body), 0x1C, 1, 0x1C, base_at, 0, base_at + len(target) + 1)
    header = bytearray(0x4C)
    header[:4] = (0x4C).to_bytes(4, "little")
    header[0x14:0x18] = (2).to_bytes(4, "little")  # it has link info
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + b"".join(n.to_bytes(4, "little") for n in fields) + body)
    return path


def test_parses_a_normal_entry_and_strips_field_codes():
    app = parse_entry("[Desktop Entry]\nType=Application\nName=Firefox\nExec=firefox %U\nIcon=firefox\n"
                      "Categories=Network;WebBrowser;\n", Path("/x/firefox.desktop"))
    assert app.id == "firefox" and app.name == "Firefox" and app.exec == "firefox"
    assert app.categories == ["Network", "WebBrowser"]


def test_skips_hidden_nodisplay_and_non_applications():
    base = "[Desktop Entry]\nName=X\nExec=x\n"
    assert parse_entry(base + "Type=Application\nNoDisplay=true\n", Path("x.desktop")) is None
    assert parse_entry(base + "Type=Application\nHidden=true\n", Path("x.desktop")) is None
    assert parse_entry(base + "Type=Link\n", Path("x.desktop")) is None


def test_ignores_localised_keys_and_keeps_base_name():
    app = parse_entry("[Desktop Entry]\nType=Application\nName=Files\nName[de]=Dateien\nExec=nautilus\n",
                      Path("org.gnome.Nautilus.desktop"))
    assert app.name == "Files"


def test_clean_exec_keeps_escaped_percent():
    assert clean_exec("app --rate 50%% %f  --x") == "app --rate 50% --x"


def app(name):
    return DesktopApp(id=name.lower(), name=name, exec=f"Start-Process {name}")


def test_find_app_ranks_exact_then_prefix_then_contains():
    cat = SystemCatalog([app("Firefox Developer"), app("Firefox"), app("Tor (Firefox)")])
    assert [a.name for a in cat.find_app("firefox")] == ["Firefox", "Firefox Developer", "Tor (Firefox)"]


def test_the_closest_names_are_offered_for_a_typo():
    cat = SystemCatalog([app("PowerPoint"), app("Word"), app("Excel")])
    assert cat.closest("powerpont", 1)[0].name == "PowerPoint"
    assert cat.closest("wrod", 1)[0].name == "Word"
    assert edit_distance("ab", "ab") == 0


def test_fingerprint_changes_when_an_app_is_added():
    a = SystemCatalog([app("Word")])
    b = SystemCatalog([app("Word"), app("Excel")])
    assert a.fingerprint() != b.fingerprint()
    assert SystemCatalog.from_json(b.to_json()).fingerprint() == b.fingerprint()


def test_start_menu_shortcuts_become_apps(tmp_path):
    (tmp_path / "Office").mkdir()
    for name in ["Office/Word.lnk", "Excel.lnk", "Uninstall Foo.lnk", "readme.txt"]:
        (tmp_path / name).write_bytes(b"")
    apps = start_menu_apps([tmp_path])
    assert sorted(a.name for a in apps) == ["Excel", "Word"]
    assert apps[0].exec.startswith("Start-Process -FilePath '")


def test_store_apps_open_through_the_apps_folder():
    json_text = ('[{"Name":"Calculator","AppID":"Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"},'
                 '{"Name":"Uninstall Foo","AppID":"foo"},{"Name":"DeepCool","AppID":"com.deepcool.creative"}]')
    apps = parse_start_apps(json_text)
    assert len(apps) == 2
    assert apps[0].id == "calculator"
    assert apps[0].exec == "Start-Process 'shell:AppsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App'"
    assert len(parse_start_apps('{"Name":"Solo","AppID":"solo"}')) == 1
    assert parse_start_apps("") == []


def test_opening_an_app_is_checked_by_its_own_program():
    # A Start-menu shortcut: the program it points to must be running.
    shortcut = checked_launch(r"Start-Process -FilePath 'C:\Start Menu\Programs\Google Chrome.lnk'")
    assert r"CreateShortcut('C:\Start Menu\Programs\Google Chrome.lnk').TargetPath" in shortcut
    # A Store app: the program its manifest names (its window is a host's).
    store = checked_launch(r"Start-Process 'shell:AppsFolder\windows.immersivecontrolpanel_cw5n1h2txyewy"
                           r"!microsoft.windows.immersivecontrolpanel'")
    assert "Get-AppxPackage -Name 'windows.immersivecontrolpanel'" in store
    assert "PackageFamilyName -eq 'windows.immersivecontrolpanel_cw5n1h2txyewy'" in store
    assert "Where-Object Id -eq 'microsoft.windows.immersivecontrolpanel'" in store
    # A program by path, and an app id that says nothing: a new window counts.
    by_path = checked_launch(r"Start-Process 'shell:AppsFolder\{1AC14E77}\notepad.exe'")
    assert r"GetFileNameWithoutExtension('{1AC14E77}\notepad.exe')" in by_path
    unknown = checked_launch(r"Start-Process 'shell:AppsFolder\Microsoft.Windows.Explorer'")
    assert "$program = @('')[0]" in unknown
    for line in (shortcut, store, by_path, unknown):
        assert "Start-Process" in line and "OPEN: " in line and "NOT CONFIRMED: " in line
        assert "Start-Sleep -Milliseconds 1000" in line, "it must still be running a second later"
        assert line.count("{") == line.count("}") and line.count("(") == line.count(")")
        assert hard_denied(line) is None


def test_a_quote_in_an_apps_name_stays_quoted():
    line = checked_launch(r"Start-Process -FilePath 'C:\Programs\Tom''s Game.lnk'")
    assert r"CreateShortcut('C:\Programs\Tom''s Game.lnk')" in line
    assert line.count("'") % 2 == 0


# ---- Windows: every installed app, not only the Start menu's ----------------------------


def test_a_programs_header_says_whether_it_has_windows(tmp_path):
    assert windowed(program(tmp_path / "app.exe")) and not windowed(program(tmp_path / "tool.exe", False))
    (tmp_path / "fake.exe").write_bytes(b"MZ" + bytes(100))
    assert not windowed(tmp_path / "fake.exe") and not windowed(tmp_path / "missing.exe")
    assert pe_subsystem(b"not a program") is None


def test_installed_names_lose_their_versions():
    for raw, name in [("7-Zip 26.01 (x64)", "7-Zip"), ("Ollama version 0.35.1", "Ollama"), ("LM Studio 0.4.25+1", "LM Studio"),
                      ("GIMP 2.10.38-1", "GIMP"), ("KMPlayer 64X (remove only)", "KMPlayer 64X"),
                      ("Microsoft Visual Studio Code (User)", "Microsoft Visual Studio Code"),
                      ("Geekbench 6", "Geekbench 6"), ("Asphalt Legends ", "Asphalt Legends")]:
        assert clean_name(raw) == name, raw


def test_an_apps_own_program_is_found_in_its_folder(tmp_path):
    folder = tmp_path / "VirtualBox"
    main = program(folder / "VirtualBox.exe")
    program(folder / "VirtualBoxVM.exe")
    program(folder / "VBoxManage.exe", False)  # a console tool
    program(folder / "unins000.exe")
    program(folder / "VirtualBoxCrashHandler.exe")
    program(folder / "plugins" / "VirtualBox.exe")  # an app's innards
    assert main_program(folder, ["VirtualBox"]) == main
    assert resembles("obs64", ["obs-studio"]) and resembles("ollama app", ["Ollama"])
    assert resembles("Code", ["Microsoft VS Code"]) and resembles("Geekbench 6", ["Geekbench 6"])
    assert not resembles("VBoxManage", ["VirtualBox"]) and not resembles("MSI_Alert_Toast", ["MSI Center"])
    assert not resembles("7zFM", ["7-Zip"])


def test_the_programs_windows_lists_as_installed(tmp_path):
    upscayl = program(tmp_path / "Upscayl" / "Upscayl.exe")
    bench = tmp_path / "Geekbench 6"
    program(bench / "Geekbench 6.exe"), program(bench / "geekbench_avx2.exe", False), program(bench / "uninstall.exe")
    ollama = tmp_path / "Ollama"
    program(ollama / "ollama app.exe"), program(ollama / "ollama.exe", False), program(ollama / "unins000.exe")
    knight = program(tmp_path / "common" / "Hollow Knight" / "hollow_knight.exe").parent
    apps = {a.name: a for a in installed_apps([
        ("{U}", {"DisplayName": "Upscayl 2.15.0", "DisplayIcon": f"{upscayl},0"}),
        # No icon: the uninstaller's folder is the app's.
        ("Geekbench 6", {"DisplayName": "Geekbench 6", "UninstallString": str(bench / "uninstall.exe")}),
        # The icon is the uninstaller's: the app's own program in its folder.
        ("Ollama_is1", {"DisplayName": "Ollama version 0.35.1", "DisplayIcon": str(ollama / "unins000.exe"),
                        "InstallLocation": str(ollama)}),
        ("Steam App 367520", {"DisplayName": "Hollow Knight", "InstallLocation": str(knight)}),
        ("{P}", {"DisplayName": "Python 3.14.7 Core Interpreter (64-bit)", "SystemComponent": 1}),
        ("{R}", {"DisplayName": "Microsoft Visual C++ 2015-2022 Redistributable (x64)", "DisplayIcon": str(upscayl)}),
        ("{K}", {"DisplayName": "Security Update for Upscayl", "ParentKeyName": "{U}"}),
        ("{N}", {"DisplayName": "Nothing On Disk"}),
    ])}
    assert sorted(apps) == ["Geekbench 6", "Hollow Knight", "Ollama", "Upscayl"]
    assert apps["Upscayl"].exec == f"Start-Process -FilePath '{upscayl}'" and apps["Upscayl"].source == "installed"
    assert apps["Geekbench 6"].exec.endswith("Geekbench 6.exe'") and apps["Ollama"].exec.endswith("ollama app.exe'")
    # A Steam game opens through Steam, and is checked by its own program.
    assert apps["Hollow Knight"].exec == "Start-Process 'steam://rungameid/367520'"
    assert apps["Hollow Knight"].program == "hollow_knight"


def test_the_programs_registered_to_start_by_name(tmp_path, monkeypatch):
    vlc = program(tmp_path / "VideoLAN" / "VLC" / "vlc.exe")
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    apps = registered_apps([
        ("vlc.exe", {"": str(vlc)}),
        ("tool.exe", {"": f'"{program(tmp_path / "Tools" / "tool.exe", False)}"'}),  # a console tool
        ("mstsc.exe", {"": str(program(tmp_path / "Windows" / "System32" / "mstsc.exe"))}),  # Windows' own
        ("gone.exe", {"": str(tmp_path / "gone.exe")}),
    ])
    assert [(a.name, a.exec, a.source) for a in apps] == [("vlc", f"Start-Process -FilePath '{vlc}'", "registered")]


def test_the_apps_in_program_files(tmp_path):
    root = tmp_path / "Program Files"
    box = program(root / "Oracle" / "VirtualBox" / "VirtualBox.exe")  # a vendor's folder
    program(root / "Oracle" / "VirtualBox" / "VBoxManage.exe", False)
    bench = program(root / "Geekbench 6" / "Geekbench 6.exe")
    obs = program(root / "obs-studio" / "bin" / "64bit" / "obs64.exe")  # below bin: still obs-studio's
    program(root / "MSI" / "MSI Center" / "Alerts" / "MSI_Alert_Toast.exe")  # named like the vendor only
    program(root / "Common Files" / "Thing" / "Thing.exe")
    program(root / "Helper" / "Helper.exe")
    apps = program_files_apps([root])
    assert sorted(a.exec for a in apps) == sorted(f"Start-Process -FilePath '{p}'" for p in (box, bench, obs))
    assert {a.source for a in apps} == {"program files"}


def test_one_app_found_twice_is_listed_once(tmp_path):
    exe = program(tmp_path / "Brave" / "brave.exe")
    link = shortcut(tmp_path / "Start Menu" / "Brave.lnk", str(exe))
    assert shortcut_target(link) == str(exe)
    start = start_menu_apps([tmp_path / "Start Menu"])
    other = program(tmp_path / "Other" / "other.exe")
    apps = merged(start, [exe_app("Brave Browser", exe, "registered"), exe_app("Other", other, "registered")])
    assert [a.name for a in apps] == ["Brave", "Other"]
    game = merged([DesktopApp("pubg battlegrounds", "PUBG BATTLEGROUNDS",
                              "Start-Process 'shell:AppsFolder\\steam://rungameid/578080'", source="apps folder")],
                  [DesktopApp("pubg: battlegrounds", "PUBG: BATTLEGROUNDS", "Start-Process 'steam://rungameid/578080'",
                              source="installed")])
    assert len(game) == 1


def test_what_the_start_menu_shows_comes_first():
    catalog = SystemCatalog([DesktopApp("word", "Word", "registered's", source="registered"),
                             DesktopApp("word", "Word", "start menu's", source="start menu")])
    assert catalog.find_app("word")[0].exec == "start menu's"


def test_a_steam_game_is_checked_by_its_own_program_and_given_longer():
    line = checked_launch("Start-Process 'steam://rungameid/367520'", "hollow_knight")
    assert "$program = @('hollow_knight')[0]" in line and "AddSeconds(15)" in line
    assert "AddSeconds(8)" in checked_launch(r"Start-Process -FilePath 'C:\Apps\app.exe'")
    assert "GetFileNameWithoutExtension('C:\\Apps\\app.exe')" in checked_launch(r"Start-Process -FilePath 'C:\Apps\app.exe'")


def test_a_catalogue_saved_before_still_loads():
    saved = '{"apps": [{"id": "word", "name": "Word", "exec": "x", "icon": "", "categories": [], "terminal": false}]}'
    app = SystemCatalog.from_json(saved).apps[0]
    assert app.name == "Word" and app.source == "" and app.program == ""
