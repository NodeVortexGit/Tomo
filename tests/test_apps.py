from pathlib import Path

from tomo.apps import DesktopApp, SystemCatalog, clean_exec, edit_distance, parse_entry, parse_start_apps, start_menu_apps


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
