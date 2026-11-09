// SPDX-License-Identifier: Apache-2.0
#include <windows.h>
#undef GetCurrentTime
#include <shobjidl.h>
#include <shellapi.h>
#include <propkey.h>
#include <propvarutil.h>
#include <dwmapi.h>
#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.ApplicationModel.DataTransfer.h>
#include <winrt/Windows.UI.h>
#include <winrt/Windows.UI.Text.h>
#include <winrt/Windows.UI.Xaml.Interop.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.System.h>
#include <winrt/Microsoft.UI.h>
#include <winrt/Microsoft.UI.Dispatching.h>
#include <winrt/Microsoft.UI.Windowing.h>
#include <winrt/Microsoft.UI.Xaml.h>
#include <winrt/Microsoft.UI.Xaml.Controls.h>
#include <winrt/Microsoft.UI.Xaml.Controls.Primitives.h>
#include <winrt/Microsoft.UI.Xaml.Markup.h>
#include <winrt/Microsoft.UI.Xaml.XamlTypeInfo.h>
#include <winrt/Microsoft.UI.Xaml.Media.h>
#include <winrt/Microsoft.UI.Xaml.Media.Imaging.h>
#include <winrt/Windows.Storage.Streams.h>
#include <winrt/Microsoft.UI.Xaml.Automation.h>
#include <winrt/Microsoft.UI.Xaml.Input.h>
#include <winrt/Microsoft.UI.Xaml.Shapes.h>
#include <microsoft.ui.xaml.window.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cwctype>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include "flare.h"
#include "Shell.xaml.h"
#include "Theme.xaml.h"
#include "resource.h"

using namespace winrt;
using namespace Microsoft::UI::Xaml;
using namespace Microsoft::UI::Xaml::Controls;
using namespace Windows::Foundation;

namespace {
int application_exit_code{};
constexpr wchar_t desktop_app_id[]=L"Flare.Desktop";
constexpr PROPERTYKEY taskbar_id_key=INIT_PKEY_AppUserModel_ID;
constexpr PROPERTYKEY taskbar_command_key=INIT_PKEY_AppUserModel_RelaunchCommand;
constexpr PROPERTYKEY taskbar_icon_key=INIT_PKEY_AppUserModel_RelaunchIconResource;
constexpr PROPERTYKEY taskbar_name_key=INIT_PKEY_AppUserModel_RelaunchDisplayNameResource;

std::wstring absolute_windows_path(std::wstring const& path) {
    DWORD capacity=GetFullPathNameW(path.c_str(),0,nullptr,nullptr);
    if(!capacity)throw_last_error();
    for(;;){
        std::wstring result(capacity,L'\0');
        DWORD length=GetFullPathNameW(path.c_str(),capacity,result.data(),nullptr);
        if(!length)throw_last_error();
        if(length<capacity){result.resize(length);return result;}
        capacity=length;
    }
}
std::wstring quote_windows_argument(std::wstring_view value) {
    std::wstring result=L"\"";size_t backslashes=0;
    for(auto character:value){
        if(character==L'\\'){++backslashes;continue;}
        result.append(character==L'"'?2*backslashes+1:backslashes,L'\\');
        result+=character;backslashes=0;
    }
    result.append(2*backslashes,L'\\');result+=L'"';return result;
}
std::wstring wide(flare_string value) {
    if (!value.data || value.length == 0) return {};
    return std::wstring(to_hstring(std::string_view(value.data, value.length)));
}
std::string utf8(flare_string value) {
    return value.data ? std::string(value.data, value.length) : std::string{};
}
std::wstring number(double value, int precision = 1) {
    if (!std::isfinite(value)) return L"—";
    std::wostringstream result;
    result << std::fixed << std::setprecision(precision) << value;
    return result.str();
}
std::wstring bytes(uint64_t value) {
    constexpr wchar_t const* units[] = { L"B", L"KiB", L"MiB", L"GiB", L"TiB" };
    double scaled = static_cast<double>(value);
    size_t unit = 0;
    while (scaled >= 1024 && unit < 4) { scaled /= 1024; ++unit; }
    return number(scaled, unit == 0 ? 0 : 1) + L" " + units[unit];
}
std::wstring count_text(uint64_t value) {
    auto text = std::to_wstring(value);
    for (auto position = static_cast<int>(text.size()) - 3; position > 0; position -= 3) text.insert(position, L",");
    return text;
}
std::wstring parameter_text(uint64_t value) {
    if(value>=1'000'000'000)return number(value/1e9,1)+L"B";
    if(value>=1'000'000)return number(value/1e6,1)+L"M";
    return count_text(value);
}
std::wstring state_name(uint32_t state) {
    switch (state) {
    case FLARE_LOADING: return L"Loading";
    case FLARE_READY: return L"Ready";
    case FLARE_DRAINING: return L"Finishing requests";
    case FLARE_STOPPING: return L"Stopping";
    case FLARE_FAILED: return L"Needs attention";
    case FLARE_RESTART_REQUIRED: return L"Restart required";
    case FLARE_REVIEW_REQUIRED: return L"Publisher review required";
    default: return L"Idle";
    }
}
using Snapshot = std::unique_ptr<flare_snapshot, decltype(&flare_snapshot_destroy)>;
struct ModelItem {
    std::string id;
    std::wstring name, source, version, path, description, error;
    uint32_t flags{};
    uint64_t size{};
    bool size_known{};
    std::wstring author, task, license, quantization, hardware, revision, revision_commit, base_model;
    uint64_t context_tokens{}, parameter_count{}, revision_bytes{};
    uint32_t rank_count{}, metadata_known{};
    bool catalog() const { return (flags & FLARE_MODEL_CATALOG) != 0; }
    bool available() const { return (flags & (FLARE_MODEL_LOCAL | FLARE_MODEL_EXTERNAL)) != 0; }
    bool recommended() const { return (flags & FLARE_MODEL_RECOMMENDED) != 0; }
};

struct DesktopApp : ApplicationT<DesktopApp, Markup::IXamlMetadataProvider> {
    DesktopApp(std::wstring config,std::wstring launcher) : config_(std::move(config)),desktop_launcher_(std::move(launcher)) {}
    ~DesktopApp() {
        clear_taskbar_properties();
        if (app_ && flare_app_is_closed(app_)) flare_app_destroy(app_);
    }
    Markup::IXamlType GetXamlType(Windows::UI::Xaml::Interop::TypeName const& type) { return metadata_.GetXamlType(type); }
    Markup::IXamlType GetXamlType(hstring const& name) { return metadata_.GetXamlType(name); }
    com_array<Markup::XmlnsDefinition> GetXmlnsDefinitions() { return metadata_.GetXmlnsDefinitions(); }

    template<typename T> T control(wchar_t const* name) const { return root_.FindName(name).as<T>(); }
    struct ModalScope {
        bool& active; bool owns;
        explicit ModalScope(bool& value):active(value),owns(!value){if(owns)active=true;}
        ~ModalScope(){release();}
        void release(){if(owns){active=false;owns=false;}}
        explicit operator bool() const{return owns;}
    };
    void text(wchar_t const* name, std::wstring const& value) { control<TextBlock>(name).Text(value); }
    void show_error(std::wstring const& message) {
        auto notice = control<InfoBar>(L"Notice");
        notice.Title(L"Flare could not complete the operation");
        notice.Message(message);
        notice.Severity(InfoBarSeverity::Error);
        notice.Margin(Thickness{0,0,0,16});
        notice.IsOpen(true);
    }
    void accepted(uint64_t operation) {
        if (!operation) show_error(wide(flare_last_error()));
        else control<InfoBar>(L"Notice").IsOpen(false);
    }
    template<typename F> void clicked(wchar_t const* name, F action) {
        control<Button>(name).Click([this, action](auto const&, auto const&) {
            try { action(); } catch (hresult_error const& error) { show_error(std::wstring(error.message())); }
            catch (std::exception const& error) { show_error(std::wstring(to_hstring(error.what()))); }
        });
    }

    void OnLaunched(LaunchActivatedEventArgs const&) {
        try { initialize_window(); }
        catch (hresult_error const& error) {
            clear_taskbar_properties();
            application_exit_code = 1;
            std::cerr << to_string(error.message()) << std::endl;
            MessageBoxW(hwnd_, error.message().c_str(), L"Flare could not start", MB_OK | MB_ICONERROR);
            if (app_) flare_app_close(app_);
            Exit();
        }
        catch (std::exception const& error) {
            clear_taskbar_properties();
            application_exit_code = 1;
            std::cerr << error.what() << std::endl;
            auto message = to_hstring(error.what());
            MessageBoxW(hwnd_, message.c_str(), L"Flare could not start", MB_OK | MB_ICONERROR);
            if (app_) flare_app_close(app_);
            Exit();
        }
    }
    void set_taskbar_property(PROPERTYKEY const& key,std::wstring const& text) {
        PROPVARIANT value{};
        check_hresult(InitPropVariantFromString(text.c_str(),&value));
        auto result=taskbar_properties_->SetValue(key,value);
        PropVariantClear(&value);check_hresult(result);
    }
    void initialize_taskbar_properties() {
        try {
            check_hresult(SHGetPropertyStoreForWindow(hwnd_,IID_PPV_ARGS(taskbar_properties_.put())));
            if(!desktop_launcher_.empty()){
                auto command=quote_windows_argument(desktop_launcher_);
                if(!config_.empty())command+=L" --config "+quote_windows_argument(config_);
                set_taskbar_property(taskbar_command_key,command);
                set_taskbar_property(taskbar_icon_key,desktop_launcher_+L",0");
                set_taskbar_property(taskbar_name_key,L"@"+desktop_launcher_+L",-103");
            }
            set_taskbar_property(taskbar_id_key,desktop_app_id);
        } catch(hresult_error const& error){
            clear_taskbar_properties();
            std::cerr<<"Taskbar relaunch metadata unavailable: "<<to_string(error.message())<<std::endl;
        }
    }
    void clear_taskbar_properties() noexcept {
        if(!taskbar_properties_)return;
        PROPVARIANT empty{};
        for(auto const& key:{taskbar_id_key,taskbar_command_key,taskbar_icon_key,taskbar_name_key}){
            auto result=taskbar_properties_->SetValue(key,empty);
            if(FAILED(result))std::cerr<<"Could not release taskbar property: "<<result<<std::endl;
        }
        taskbar_properties_=nullptr;
    }
    void initialize_window() {
        Resources().MergedDictionaries().Append(XamlControlsResources());
        theme_resources_ = Markup::XamlReader::Load(theme_xaml).as<ResourceDictionary>();
        Resources().MergedDictionaries().Append(theme_resources_);
        window_ = Window();
        window_.Title(L"Flare — Local inference");
        root_ = Markup::XamlReader::Load(shell_xaml).as<FrameworkElement>();
        root_.RequestedTheme(ElementTheme::Dark);
        window_.Content(root_);
        window_.ExtendsContentIntoTitleBar(true);
        window_.SetTitleBar(control<Grid>(L"NativeTitleBar"));
        check_hresult(window_.as<IWindowNative>()->get_WindowHandle(&hwnd_));
        initialize_taskbar_properties();
        if (SUCCEEDED(CoCreateInstance(CLSID_TaskbarList, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(taskbar_.put())))) taskbar_->HrInit();
        BOOL dark = TRUE;
        DwmSetWindowAttribute(hwnd_, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark, sizeof(dark));
        auto app_window = window_.AppWindow();
        auto scale = static_cast<double>(GetDpiForWindow(hwnd_)) / 96.0;
        int32_t width = static_cast<int32_t>(1200 * scale), height = static_cast<int32_t>(1000 * scale);
        MONITORINFO monitor{ sizeof(MONITORINFO) };
        if (GetMonitorInfoW(MonitorFromWindow(hwnd_, MONITOR_DEFAULTTONEAREST), &monitor)) {
            auto work_width = monitor.rcWork.right - monitor.rcWork.left;
            auto work_height = monitor.rcWork.bottom - monitor.rcWork.top;
            width = std::min(width, static_cast<int32_t>(work_width - 32));
            height = std::min(height, static_cast<int32_t>(work_height - 32));
            app_window.MoveAndResize({ monitor.rcWork.left + (work_width - width) / 2,
                monitor.rcWork.top + (work_height - height) / 2, width, height });
        } else app_window.Resize({ width, height });
        if (auto presenter = app_window.Presenter().try_as<Microsoft::UI::Windowing::OverlappedPresenter>()) {
            presenter.PreferredMinimumWidth(box_value(std::min(width, static_cast<int32_t>(840 * scale))).as<IReference<int32_t>>());
            presenter.PreferredMinimumHeight(box_value(std::min(height, static_cast<int32_t>(600 * scale))).as<IReference<int32_t>>());
        }
        app_window.Closing([this](auto const&, Microsoft::UI::Windowing::AppWindowClosingEventArgs const& args) {
            if (closed_) return;
            args.Cancel(true);
            begin_close();
        });
        auto navigation = control<NavigationView>(L"Navigation");
        navigation.SelectionChanged([this](auto const&, NavigationViewSelectionChangedEventArgs const& args) {
            if (args.IsSettingsSelected()) navigate(L"Settings");
            else if (auto item = args.SelectedItem().try_as<NavigationViewItem>()) navigate(std::wstring(unbox_value<hstring>(item.Tag())));
        });
        navigation.SelectedItem(navigation.MenuItems().GetAt(0));
        clicked(L"ChooseModelButton", [this] { choose_model(); });
        clicked(L"HomeAllModelsButton",[this]{select_source(false);select_page(1);});
        clicked(L"RefreshButton", [this] { if(discover_tab_)search_catalog();else accepted(flare_app_refresh(app_)); });
        clicked(L"StopButton", [this] { stop_serving(); });
        clicked(L"CancelSwitchButton", [this] { if (current_operation_) accepted(flare_app_cancel(app_, current_operation_)); });
        for (auto name : { L"AddModelButton", L"EmptyAddButton", L"AddDirectButton",L"HomeAddButton",L"HomeEmptyAddButton" }) clicked(name, [this] { add_model(); });
        clicked(L"SearchButton", [this] { search_catalog(); });
        clicked(L"DiscoverTab", [this] { select_source(true); });
        clicked(L"LibraryTab", [this] { select_source(false); });
        clicked(L"ViewDownloadsButton", [this] { select_page(2); });
        clicked(L"DownloadsFindButton", [this] { select_source(true); select_page(1); });
        clicked(L"CopyEndpointButton",[this]{if(endpoint_.empty())return;copy_text(endpoint_+L"/v1");copied_endpoint_=endpoint_;endpoint_copy_until_=std::chrono::steady_clock::now()+std::chrono::seconds(2);text(L"EndpointCopyStatus",L"Copied");});
        initialize_chat();
        clicked(L"RetryModelButton", [this] { if (!pending_model_id_.empty()) activate(pending_model_id_, pending_model_name_); });
        for (auto name : { L"StorageButton", L"DownloadStorageButton" }) clicked(name, [this] { storage_dialog(); });
        clicked(L"AccountButton", [this] { account_dialog(); });
        initialize_runtime_settings();
        control<TextBox>(L"ModelSearch").TextChanged([this](auto const&, auto const&) { model_fingerprint_.clear(); render_model_cards(); });
        control<TextBox>(L"ModelSearch").KeyDown([this](auto const&, auto const& args) { if (args.Key() == Windows::System::VirtualKey::Enter && discover_tab_) search_catalog(); });
        control<GridView>(L"ModelList").SizeChanged([this](auto const&, auto const&) { resize_cards(); });
        for(auto name:{L"PrefillChart",L"GenerationChart",L"VerifyChart",L"DraftChart"})control<Canvas>(name).SizeChanged([this](auto const&, auto const&) { draw_chart(); });
        control<ComboBox>(L"DensityChoice").SelectionChanged([this](auto const&, auto const&) { apply_density(); });
        control<CheckBox>(L"CompatibleFilter").Checked([this](auto const&,auto const&){filter_incompatible_=true;render_model_cards();});
        control<CheckBox>(L"CompatibleFilter").Unchecked([this](auto const&,auto const&){filter_incompatible_=false;render_model_cards();});
        root_.ActualThemeChanged([this](auto const&, auto const&) { apply_caption_theme(); });
        root_.AllowDrop(true);
        root_.DragOver([](auto const&, DragEventArgs const& args) { if (args.DataView().Contains(Windows::ApplicationModel::DataTransfer::StandardDataFormats::StorageItems())) args.AcceptedOperation(Windows::ApplicationModel::DataTransfer::DataPackageOperation::Link); });
        root_.Drop([this](auto const&, DragEventArgs const& args) { drop_core(args); });
        app_ = flare_app_create(reinterpret_cast<uint16_t const*>(config_.data()), config_.size());
        if (!app_) throw hresult_error(E_FAIL, hstring(wide(flare_last_error())));
        timer_ = root_.DispatcherQueue().CreateTimer();
        timer_.Interval(std::chrono::milliseconds(500));
        timer_.Tick([this](auto const&, auto const&) { poll(); });
        timer_.Start();
        window_.Activate();
        load_title_icon();
        auto icon_dpi=GetDpiForWindow(hwnd_);
        for(auto size:{ICON_SMALL,ICON_BIG}) {
            auto metric=size==ICON_SMALL?SM_CXSMICON:SM_CXICON;
            auto icon=LoadImageW(GetModuleHandleW(nullptr),MAKEINTRESOURCEW(IDI_FLARE_APP),IMAGE_ICON,GetSystemMetricsForDpi(metric,icon_dpi),GetSystemMetricsForDpi(metric+1,icon_dpi),LR_SHARED);
            if(icon)SendMessageW(hwnd_,WM_SETICON,size,reinterpret_cast<LPARAM>(icon));
        }
        apply_caption_theme();
        navigate(L"Overview");
        poll();
    }

    fire_and_forget load_title_icon() {
        auto lifetime=get_strong();
        try {
            auto module=GetModuleHandleW(nullptr);
            auto resource=FindResourceW(module,MAKEINTRESOURCEW(IDB_FLARE_TITLE),RT_RCDATA);
            if(!resource)throw_last_error();
            auto handle=LoadResource(module,resource);auto bytes=static_cast<uint8_t const*>(LockResource(handle));auto length=SizeofResource(module,resource);
            if(!bytes||!length)throw hresult_error(E_FAIL,L"Embedded title icon is empty");
            Windows::Storage::Streams::InMemoryRandomAccessStream stream;
            Windows::Storage::Streams::DataWriter writer(stream);
            writer.WriteBytes(array_view<uint8_t const>(bytes,bytes+length));
            co_await writer.StoreAsync();writer.DetachStream();stream.Seek(0);
            Microsoft::UI::Xaml::Media::Imaging::BitmapImage bitmap;
            co_await bitmap.SetSourceAsync(stream);
            if(!closing_)control<Image>(L"TitleIcon").Source(bitmap);
        } catch(hresult_error const& error){std::cerr<<"Flare title icon: "<<to_string(error.message())<<std::endl;}
    }
    void select_page(uint32_t index) {
        auto navigation = control<NavigationView>(L"Navigation");
        navigation.SelectedItem(navigation.MenuItems().GetAt(index));
    }
    void navigate(std::wstring const& page) {
        page_ = page;
        for (auto name : { L"Overview", L"Models", L"Downloads", L"Chat", L"Settings" }) {
            auto element_name = std::wstring(name) + L"Page";
            control<FrameworkElement>(element_name.c_str()).Visibility(page == name ? Visibility::Visible : Visibility::Collapsed);
        }
        control<Grid>(L"PageHeader").Visibility(page == L"Overview" || page == L"Chat" ? Visibility::Collapsed : Visibility::Visible);
        control<Button>(L"AddModelButton").Visibility(page == L"Models" ? Visibility::Visible : Visibility::Collapsed);
        control<Button>(L"RefreshButton").Visibility(page == L"Models" ? Visibility::Visible : Visibility::Collapsed);
        text(L"PageTitle", page);
        if (page == L"Overview") draw_chart();
        if (page == L"Models" && discover_tab_ && !catalog_requested_ && app_) search_catalog();
    }
    void open_core(bool folder, bool storage) {
        com_ptr<IFileOpenDialog> picker;
        check_hresult(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(picker.put())));
        DWORD options{};
        check_hresult(picker->GetOptions(&options));
        check_hresult(picker->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | (folder ? FOS_PICKFOLDERS : FOS_FILEMUSTEXIST)));
        check_hresult(picker->SetTitle(storage ? L"Choose model storage" : folder ? L"Open prepared core folder" : L"Open prepared core"));
        if (!folder) {
            COMDLG_FILTERSPEC filters[] = { { L"Flare prepared cores", L"*.safetensors" }, { L"All files", L"*.*" } };
            check_hresult(picker->SetFileTypes(2, filters));
        }
        auto result = picker->Show(hwnd_);
        if (result == HRESULT_FROM_WIN32(ERROR_CANCELLED)) return;
        check_hresult(result);
        com_ptr<IShellItem> item;
        check_hresult(picker->GetResult(item.put()));
        PWSTR path{};
        check_hresult(item->GetDisplayName(SIGDN_FILESYSPATH, &path));
        std::unique_ptr<wchar_t, decltype(&CoTaskMemFree)> path_owner(path, CoTaskMemFree);
        auto length = wcslen(path);
        accepted(storage ? flare_app_set_model_directory(app_, reinterpret_cast<uint16_t*>(path), length) :
            flare_app_open_model(app_, reinterpret_cast<uint16_t*>(path), length));
        if (!storage) { select_source(false); select_page(1); }
    }
    void download_repository(std::wstring const& value) {
        auto repository = to_string(value);
        auto operation = flare_app_download(app_, repository.data(), repository.size());
        accepted(operation);
        if (operation) select_page(2);
    }
    fire_and_forget activate(std::string model, std::wstring name, bool reload=false) {
        auto lifetime = get_strong();
        ModalScope modal_scope{modal_open_};if(!modal_scope)co_return;
        try {
            if (session_state_ == FLARE_READY) {
                ContentDialog dialog;
                dialog.XamlRoot(root_.XamlRoot());
                dialog.Title(box_value(reload?L"Reload model?":L"Switch active model?"));
                dialog.Content(box_value(L"Flare will finish accepted requests, release the active model, and load " + name +
                    L". The endpoint is unavailable while the replacement loads."));
                dialog.PrimaryButtonText(reload?L"Finish requests and reload":L"Finish requests and switch");
                dialog.CloseButtonText(L"Cancel");
                dialog.DefaultButton(ContentDialogButton::Close);
                if (co_await dialog.ShowAsync() != ContentDialogResult::Primary) co_return;
            }
            if (closing_) co_return;
            auto operation = flare_app_activate(app_, model.data(), model.size());
            accepted(operation);
            if (operation) { can_change_model_=false;control<Button>(L"ChooseModelButton").IsEnabled(false);pending_model_id_ = model; pending_model_name_ = name; observed_steps_.clear(); select_page(0);render_model_cards();render_home_models(); }
        } catch (hresult_error const& error) { show_error(std::wstring(error.message())); }
    }
    fire_and_forget stop_serving() {
        auto lifetime = get_strong();
        ModalScope modal_scope{modal_open_};if(!modal_scope)co_return;
        try {
            ContentDialog dialog;
            dialog.XamlRoot(root_.XamlRoot());
            dialog.Title(box_value(L"Stop serving?"));
            dialog.Content(box_value(L"Flare will stop the local server and release the model. Active requests may be interrupted."));
            dialog.PrimaryButtonText(L"Stop serving");
            dialog.CloseButtonText(L"Keep serving");
            dialog.DefaultButton(ContentDialogButton::Close);
            if (co_await dialog.ShowAsync() == ContentDialogResult::Primary && !closing_) accepted(flare_app_stop(app_));
        } catch (hresult_error const& error) { show_error(std::wstring(error.message())); }
    }

    TextBlock label(std::wstring const& value, double size = 14, bool secondary = false) {
        TextBlock result;
        result.Text(value);
        result.FontSize(size);
        result.TextWrapping(TextWrapping::Wrap);
        result.IsTextSelectionEnabled(true);
        if (secondary) result.Style(theme_resources_.Lookup(box_value(L"Secondary")).as<Style>());
        return result;
    }
    Border card(StackPanel const& content) {
        Border result;
        result.Style(theme_resources_.Lookup(box_value(L"Card")).as<Style>());
        result.Padding(Thickness{card_padding_,card_padding_,card_padding_,card_padding_});
        result.Child(content);
        return result;
    }
    Border model_mark(std::wstring const& name) {
        auto result=Markup::XamlReader::Load(L"<Border xmlns='http://schemas.microsoft.com/winfx/2006/xaml/presentation' Width='36' Height='36' CornerRadius='4' Background='{ThemeResource SelectedBrush}'><TextBlock FontSize='20' Foreground='{ThemeResource AppAccentBrush}' HorizontalAlignment='Center' VerticalAlignment='Center'/></Border>").as<Border>();
        result.Child().as<TextBlock>().Text(name.empty()?L"F":name.substr(0,1));return result;
    }
    ContentDialog dialog(std::wstring const& title) {
        ContentDialog result; result.XamlRoot(root_.XamlRoot()); result.RequestedTheme(root_.ActualTheme());result.Title(label(title,22));
        result.CloseButtonText(L"Close"); result.DefaultButton(ContentDialogButton::None); return result;
    }
    Button dialog_choice(std::wstring const& title,std::wstring const& detail) {
        Button result; result.HorizontalAlignment(HorizontalAlignment::Stretch); result.HorizontalContentAlignment(HorizontalAlignment::Stretch); result.Padding(Thickness{12,12,12,12});
        Automation::AutomationProperties::SetName(result,title);
        StackPanel content; content.Spacing(3); auto heading=label(title,14); heading.FontWeight(Windows::UI::Text::FontWeights::SemiBold()); content.Children().Append(heading); content.Children().Append(label(detail,12,true)); result.Content(content); return result;
    }
    fire_and_forget add_model() {
        auto lifetime=get_strong();
        ModalScope modal_scope{modal_open_};if(!modal_scope)co_return;
        try {
            auto modal=dialog(L"Add a model"); modal.CloseButtonText(L"Cancel");
            StackPanel body; body.Spacing(10);
            auto choice=std::make_shared<int>(-1); weak_ref<ContentDialog> weak{modal};
            const wchar_t* titles[]={L"Discover models",L"Hugging Face repository",L"Open core file…",L"Open core folder…"};
            const wchar_t* descriptions[]={L"Browse Flare models on Hugging Face",L"Paste a repository name or link",L"Select a prepared .safetensors core",L"Select a complete prepared core folder"};
            for(int i=0;i<4;++i) { auto button=dialog_choice(titles[i],descriptions[i]); button.Click([choice,weak,i](auto const&,auto const&){*choice=i;if(auto shown=weak.get())shown.Hide();}); body.Children().Append(button); }
            modal.Content(body); co_await modal.ShowAsync();modal_scope.release();
            if(closing_)co_return;
            if(*choice==0) {select_source(true);select_page(1);} else if(*choice==1)add_repository(); else if(*choice==2)open_core(false,false); else if(*choice==3)open_core(true,false);
        } catch(hresult_error const& error) {show_error(std::wstring(error.message()));}
    }
    fire_and_forget add_repository() {
        auto lifetime=get_strong();
        ModalScope modal_scope{modal_open_};if(!modal_scope)co_return;
        try {
            auto modal=dialog(L"Add from Hugging Face"); modal.PrimaryButtonText(L"Find prepared cores"); modal.SecondaryButtonText(L"Download repository");
            StackPanel body; body.Spacing(12); TextBox input; input.Header(box_value(L"Repository name or link")); input.PlaceholderText(L"publisher/core or a Hugging Face URL"); body.Children().Append(input);
            body.Children().Append(label(L"Search discovery metadata, or download an exact prepared Flare repository. Core checks and publisher review happen before use.",12,true)); modal.Content(body);
            auto result=co_await modal.ShowAsync();modal_scope.release(); if(closing_)co_return;
            if(result==ContentDialogResult::Primary) {select_source(true);select_page(1);control<TextBox>(L"ModelSearch").Text(input.Text());search_catalog();}
            else if(result==ContentDialogResult::Secondary) download_repository(std::wstring(input.Text()));
        } catch(hresult_error const& error) {show_error(std::wstring(error.message()));}
    }
    fire_and_forget choose_model() {
        auto lifetime=get_strong();
        ModalScope modal_scope{modal_open_};if(!modal_scope)co_return;
        try {
            auto modal=dialog(L"Choose a model"); modal.CloseButtonText(L"Cancel"); StackPanel body; body.Spacing(9); body.Children().Append(label(L"Downloaded cores on this PC.",14,true));
            auto choice=std::make_shared<std::string>(); weak_ref<ContentDialog> weak{modal};
            for(auto const& model:models_) if(model.available()&&!model.catalog()) {auto button=dialog_choice(model.name,model.flags&FLARE_MODEL_ACTIVE?L"In use":model.size_known?bytes(model.size):L"Inspect before loading"); button.Click([choice,weak,id=model.id](auto const&,auto const&){*choice=id;if(auto shown=weak.get())shown.Hide();}); body.Children().Append(button);}
            if(body.Children().Size()==1) {select_source(false);select_page(1);co_return;}
            modal.Content(body); co_await modal.ShowAsync();modal_scope.release(); if(closing_)co_return;
            for(auto const& model:models_) if(!model.catalog()&&model.id==*choice && !(model.flags&FLARE_MODEL_ACTIVE)) {activate(model.id,model.name);break;}
        } catch(hresult_error const& error) {show_error(std::wstring(error.message()));}
    }
    void copy_text(std::wstring const& value) {
        Windows::ApplicationModel::DataTransfer::DataPackage data; data.SetText(value); Windows::ApplicationModel::DataTransfer::Clipboard::SetContent(data);
    }
    fire_and_forget storage_dialog() {
        auto lifetime=get_strong();
        ModalScope modal_scope{modal_open_};if(!modal_scope)co_return;
        try {
            auto modal=dialog(L"Core storage");StackPanel body;body.Spacing(12);body.Children().Append(label(model_directory_,13));
            size_t local=0;for(auto const& model:models_)if(model.available()&&!model.catalog())++local;
            body.Children().Append(label(std::to_wstring(local)+(local==1?L" core in this library":L" cores in this library"),14));
            body.Children().Append(label(L"Downloads use this directory. External references keep their original files. Choosing another directory does not move existing cores.",13,true));
            modal.Content(body);modal.PrimaryButtonText(L"Choose directory…");
            if(co_await modal.ShowAsync()==ContentDialogResult::Primary&&!closing_)open_core(true,true);
        }catch(hresult_error const& error){show_error(std::wstring(error.message()));}
    }
    fire_and_forget account_dialog() {
        auto lifetime=get_strong();
        ModalScope modal_scope{modal_open_};if(!modal_scope)co_return;try {auto modal=dialog(L"Hugging Face access");modal.Content(label(L"Public cores can be discovered without an account. Private and gated repositories require Hugging Face credentials configured for Flare. Publisher trust and license review remain separate from repository access.",14,true));co_await modal.ShowAsync();}catch(hresult_error const& error){show_error(std::wstring(error.message()));}
    }
    void select_source(bool discover) {
        discover_tab_=discover;
        control<TextBox>(L"ModelSearch").Text(L"");
        control<TextBox>(L"ModelSearch").PlaceholderText(discover?L"Search Hugging Face models":L"Search downloaded models");
        control<Button>(L"SearchButton").Visibility(discover?Visibility::Visible:Visibility::Collapsed);
        control<Border>(L"DiscoverUnderline").Visibility(discover?Visibility::Visible:Visibility::Collapsed);
        control<Border>(L"LibraryUnderline").Visibility(discover?Visibility::Collapsed:Visibility::Visible);
        control<Button>(L"DiscoverTab").FontWeight(discover?Windows::UI::Text::FontWeights::SemiBold():Windows::UI::Text::FontWeights::Normal());
        control<Button>(L"LibraryTab").FontWeight(discover?Windows::UI::Text::FontWeights::Normal():Windows::UI::Text::FontWeights::SemiBold());
        control<TextBlock>(L"CatalogState").Visibility(discover?Visibility::Visible:Visibility::Collapsed);
        render_model_cards();
        if(discover && !catalog_requested_ && app_)search_catalog();
    }
    void search_catalog() {
        auto query=to_string(control<TextBox>(L"ModelSearch").Text());
        accepted(flare_app_discover(app_,query.data(),query.size()));
        catalog_requested_=true;
    }
    void apply_caption_theme() {
        if(!window_ || !root_) return;
        HIGHCONTRASTW accessibility{sizeof(HIGHCONTRASTW)};
        bool high_contrast=SystemParametersInfoW(SPI_GETHIGHCONTRAST,sizeof(accessibility),&accessibility,0)&&(accessibility.dwFlags&HCF_HIGHCONTRASTON);
        BOOL dark=!high_contrast;
        DwmSetWindowAttribute(hwnd_,DWMWA_USE_IMMERSIVE_DARK_MODE,&dark,sizeof(dark));
        auto bar=window_.AppWindow().TitleBar();
        auto transparent=box_value(Windows::UI::Color{0,0,0,0}).as<IReference<Windows::UI::Color>>();
        bar.ButtonBackgroundColor(transparent); bar.ButtonInactiveBackgroundColor(transparent);
        auto system_text=GetSysColor(COLOR_WINDOWTEXT);
        auto foreground=box_value(high_contrast?Windows::UI::Color{255,GetRValue(system_text),GetGValue(system_text),GetBValue(system_text)}:Windows::UI::Color{255,245,245,245}).as<IReference<Windows::UI::Color>>();
        bar.ButtonForegroundColor(foreground);
        draw_chart();
    }
    void apply_density() {
        bool compact=control<ComboBox>(L"DensityChoice").SelectedIndex()==1;
        card_padding_=compact?12.0:16.0;
        auto workspace_padding=compact?16.0:24.0;
        control<Grid>(L"Workspace").Padding(Thickness{workspace_padding,workspace_padding,workspace_padding,workspace_padding});
        for(auto name:{L"MetricsCard",L"KVCacheCard",L"HostCacheCard",L"LoadingCard",L"MemoryCard",L"DownloadSummaryCard",L"DownloadsStorageCard",L"DisplayCard",L"SettingsStorageCard",L"ServerSettingsCard",L"MemorySettingsCard",L"LoadSettingsCard",L"PrefixSettingsCard",L"AdvancedSettingsCard"}) control<Border>(name).Padding(Thickness{card_padding_,card_padding_,card_padding_,card_padding_});
        render_model_cards(); resize_cards();render_home_models();
        save_density();
    }
    void save_density() {
        if(!app_||applying_preferences_)return;
        auto compact=control<ComboBox>(L"DensityChoice").SelectedIndex()==1;
        auto operation=flare_app_set_appearance(app_,2,compact?1u:0u);
        accepted(operation);if(operation){preferences_pending_=true;pending_compact_=compact;}
    }
    void update_density(flare_status const& value) {
        if(preferences_pending_&&value.appearance_compact!=static_cast<uint32_t>(pending_compact_))return;
        preferences_pending_=false;
        if(preferences_initialized_&&persisted_compact_==(value.appearance_compact!=0))return;
        applying_preferences_=true;persisted_compact_=value.appearance_compact!=0;
        control<ComboBox>(L"DensityChoice").SelectedIndex(persisted_compact_?1:0);apply_density();
        applying_preferences_=false;preferences_initialized_=true;
    }
    void draw_chart() {
        if(!root_ || page_!=L"Overview" || IsIconic(hwnd_) || !IsWindowVisible(hwnd_))return;
        using Point=flare_dashboard_point;
        draw_metric_chart(L"Prefill",L"PrefillSeriesBrush",&Point::prefill_tokens_per_second);
        draw_metric_chart(L"Generation",L"AppAccentBrush",&Point::generated_tokens_per_second);
        draw_metric_chart(L"Verify",L"VerifySeriesBrush",&Point::verify_steps_per_second);
        draw_metric_chart(L"Draft",L"AppAccentBrush",&Point::acceptance,FLARE_DASHBOARD_ACCEPTANCE);
    }
    void draw_metric_chart(std::wstring const& name,wchar_t const* brush,double flare_dashboard_point::* field,uint32_t known=0) {
        auto canvas=control<Canvas>((name+L"Chart").c_str());canvas.Children().Clear();
        double maximum=1;
        for(auto const& point:dashboard_points_) {
            if((!known||(point.known_fields&known))&&std::isfinite(point.*field))maximum=std::max(maximum,point.*field*1.15);
            if(known&&(point.known_fields&FLARE_DASHBOARD_DEPTH)&&std::isfinite(point.depth))maximum=std::max(maximum,point.depth*1.15);
        }
        auto axis=maximum>=1000?number(maximum/1000,1)+L"k":number(maximum,maximum>=10?0:1);
        text((name+L"ChartMax").c_str(),dashboard_.available?axis:L"—");
        if(canvas.ActualWidth()<=0||canvas.ActualHeight()<=0)return;
        auto series=[&](double flare_dashboard_point::* member,uint32_t flag,wchar_t const* color,bool dashed) {
            Shapes::Polyline line{nullptr};Media::PointCollection points{nullptr};
            for(auto const& point:dashboard_points_) {
                double value=point.*member;
                if((flag&&!(point.known_fields&flag))||!std::isfinite(value)){line=nullptr;continue;}
                if(!line) {
                    line=Shapes::Polyline();points=Media::PointCollection();
                    line.Stroke(theme_resources_.Lookup(box_value(color)).as<Media::Brush>());line.StrokeThickness(1.75);
                    if(dashed){Media::DoubleCollection dashes;dashes.Append(3);dashes.Append(2);line.StrokeDashArray(dashes);}
                    line.Points(points);canvas.Children().Append(line);
                }
                auto age=dashboard_.uptime_seconds-point.elapsed_seconds;
                float x=static_cast<float>(std::clamp(1-age/120.0,0.0,1.0)*canvas.ActualWidth());
                float y=static_cast<float>(1+(1-std::clamp(value/maximum,0.0,1.0))*(canvas.ActualHeight()-2));
                points.Append({x,y});
            }
        };
        series(field,known,brush,false);
        if(known)series(&flare_dashboard_point::depth,FLARE_DASHBOARD_DEPTH,L"SecondaryTextBrush",true);
        Automation::AutomationProperties::SetName(canvas,name+L" over the last 120 seconds");
    }
    fire_and_forget drop_core(DragEventArgs args) {
        auto lifetime=get_strong();
        try {
            auto items=co_await args.DataView().GetStorageItemsAsync();
            if(items.Size()!=1) { show_error(L"Drop one prepared core file or folder at a time."); co_return; }
            auto path=items.GetAt(0).Path();
            if(!closing_) { accepted(flare_app_open_model(app_,reinterpret_cast<uint16_t const*>(path.c_str()),path.size())); select_source(false); select_page(1); }
        } catch(hresult_error const& error) { show_error(std::wstring(error.message())); }
    }
    void render_models(flare_snapshot const* snapshot) {
        auto count = flare_snapshot_model_count(snapshot);
        std::wstring fingerprint = can_change_model_ ? L"enabled;" : L"disabled;";
        std::vector<ModelItem> models;
        for (size_t i = 0; i < std::min(count, size_t{200}); ++i) {
            flare_model view{};
            if (!flare_snapshot_model(snapshot, i, &view)) continue;
            models.push_back({ utf8(view.id), wide(view.name), wide(view.source), wide(view.version), wide(view.path), wide(view.description), wide(view.error), view.flags, view.logical_bytes, view.size_known != 0 });
            auto& item=models.back();item.author=wide(view.author);item.task=wide(view.task);item.license=wide(view.license);item.quantization=wide(view.quantization);item.hardware=wide(view.hardware);item.revision=wide(view.revision);item.revision_commit=wide(view.revision_commit);item.base_model=wide(view.base_model);
            item.context_tokens=view.context_tokens;item.parameter_count=view.parameter_count;item.revision_bytes=view.revision_bytes;item.rank_count=view.rank_count;item.metadata_known=view.metadata_known;
            for (auto field : { view.id, view.name, view.source, view.version, view.path, view.description, view.error }) fingerprint += std::to_wstring(field.length) + L":" + wide(field);
            for (auto field : { view.author, view.task, view.license, view.quantization, view.hardware, view.revision, view.revision_commit, view.base_model }) fingerprint += std::to_wstring(field.length) + L":" + wide(field);
            fingerprint += L"/"+std::to_wstring(view.context_tokens)+L"/"+std::to_wstring(view.parameter_count)+L"/"+std::to_wstring(view.revision_bytes)+L"/"+std::to_wstring(view.rank_count)+L"/"+std::to_wstring(view.metadata_known);
            fingerprint += L"/" + std::to_wstring(view.flags) + L"/" + std::to_wstring(view.logical_bytes) + L"/" + std::to_wstring(view.size_known) + L";";
        }
        if (fingerprint == model_fingerprint_) return;
        model_fingerprint_ = std::move(fingerprint);
        models_ = std::move(models);
        render_model_cards();
        render_home_models();
    }
    void model_fact(StackPanel const& body, std::wstring const& key, std::wstring const& value) {
        Grid row;row.ColumnSpacing(12);ColumnDefinition caption;caption.Width(GridLength{76,GridUnitType::Pixel});row.ColumnDefinitions().Append(caption);row.ColumnDefinitions().Append(ColumnDefinition());
        row.Children().Append(label(key,12,true));auto text=label(value,12);text.MaxLines(2);Grid::SetColumn(text,1);row.Children().Append(text);body.Children().Append(row);
    }
    void render_model_cards() {
        if (!root_) return;
        auto list = control<GridView>(L"ModelList");
        list.Items().Clear();
        auto recommended=control<StackPanel>(L"RecommendedModelList");recommended.Children().Clear();
        auto query = std::wstring(control<TextBox>(L"ModelSearch").Text());
        std::transform(query.begin(), query.end(), query.begin(), [](wchar_t c){return static_cast<wchar_t>(std::towlower(c));});
        size_t count = 0,recommended_count=0,other_count=0;
        for (auto const& model : models_) {
            if (model.catalog() != discover_tab_) continue;
            if(filter_incompatible_) {auto match=compatibilities_.find(model.id);if(match!=compatibilities_.end()&&match->second==FLARE_COMPATIBILITY_INCOMPATIBLE)continue;}
            auto searchable = model.name + L" " + model.description + L" " + model.source + L" " + model.author + L" " + model.task + L" " + model.quantization;
            std::transform(searchable.begin(), searchable.end(), searchable.begin(), [](wchar_t c){return static_cast<wchar_t>(std::towlower(c));});
            if (!query.empty() && searchable.find(query) == std::wstring::npos) continue;
            ++count;
            if(discover_tab_&&model.recommended()){
                recommended.Children().Append(home_model_card(model,L"ModelsRecommendedAction"+std::to_wstring(recommended_count++),true));
                continue;
            }
            ++other_count;
            StackPanel content;
            content.Spacing(10);
            Grid hero;
            hero.Height(62);
            hero.ColumnDefinitions().Append(ColumnDefinition());
            ColumnDefinition name_column; name_column.Width(GridLength{ 1, GridUnitType::Star }); hero.ColumnDefinitions().Append(name_column);
            hero.ColumnDefinitions().GetAt(0).Width(GridLength{44,GridUnitType::Pixel});
            hero.Children().Append(model_mark(model.name));
            if(model.recommended()){ColumnDefinition badge_column;badge_column.Width(GridLength{1,GridUnitType::Auto});hero.ColumnDefinitions().Append(badge_column);auto badge=recommendation_badge();Grid::SetColumn(badge,2);hero.Children().Append(badge);}
            StackPanel heading; heading.Spacing(2); heading.Margin(Thickness{10,0,0,0}); Grid::SetColumn(heading,1);
            Button title; title.Style(theme_resources_.Lookup(box_value(L"TextAction")).as<Style>()); title.Padding(Thickness{0});
            auto name=label(model.name,16); name.FontWeight(Windows::UI::Text::FontWeights::SemiBold()); name.MaxLines(2); title.Content(name);
            title.Click([this, model](auto const&,auto const&) { model_details(model); });
            heading.Children().Append(title); heading.Children().Append(label(model.author.empty()?(model.catalog()?L"Hugging Face":L"On this PC"):model.author,12,true)); hero.Children().Append(heading); content.Children().Append(hero);
            auto summary=model.task;
            if(summary==L"text-generation")summary=L"Text generation";
            else if(summary==L"image-text-to-text")summary=L"Text + vision";
            if(!model.quantization.empty())summary+=(summary.empty()?L"":L" · ")+model.quantization;
            if(model.metadata_known&FLARE_MODEL_PARAMETERS)summary+=(summary.empty()?L"":L" · ")+parameter_text(model.parameter_count)+L" parameters";
            if(!summary.empty()){auto caption=label(summary,12,true);caption.MaxLines(2);content.Children().Append(caption);}
            StackPanel facts;facts.Spacing(5);facts.MinHeight(85);
            if(model.metadata_known&FLARE_MODEL_CONTEXT)model_fact(facts,L"Context",count_text(model.context_tokens)+L" tokens");
            if(model.available()&&model.size_known)model_fact(facts,L"Core data",bytes(model.size));
            else if(model.metadata_known&FLARE_MODEL_REVISION_SIZE)model_fact(facts,L"Hub files",bytes(model.revision_bytes)+L" · "+model.revision);
            auto hardware=model.hardware;if(model.metadata_known&FLARE_MODEL_RANKS)hardware+=(hardware.empty()?L"":L" · ")+std::to_wstring(model.rank_count)+(model.rank_count==1?L" GPU":L" GPUs");
            if(!hardware.empty())model_fact(facts,model.available()?L"Core target":L"Builds",hardware);
            if(!model.license.empty())model_fact(facts,L"License",model.license);
            ToolTipService::SetToolTip(facts,box_value(model.available()?L"Context and hardware are recorded in the installed core. Core data counts logical payload bytes; storage and GPU memory use can differ. Loading performs compatibility checks.":L"Metadata reported by the repository. Hub files is the complete revision size, not GPU memory or an exact download estimate."));
            content.Children().Append(facts);
            if(model.flags&FLARE_MODEL_ACTIVE)content.Children().Append(label(L"● In use",12));
            else if(!model.error.empty())content.Children().Append(label(L"Needs attention",12));
            else if(model.available())content.Children().Append(label(L"Downloaded",12));
            StackPanel actions; actions.Orientation(Orientation::Horizontal); actions.Spacing(8);
            Button details; details.Content(box_value(L"Details")); details.Click([this,model](auto const&,auto const&) { model_details(model); }); actions.Children().Append(details);
            Button action; action.Content(box_value(model.flags&FLARE_MODEL_ACTIVE?L"In use":model.available()?L"Run model":L"Download"));
            action.IsEnabled((!model.available()||can_change_model_) && !(model.flags&FLARE_MODEL_ACTIVE) && model.error.empty());
            Automation::AutomationProperties::SetName(action,(model.available()?L"Use ":L"Download ")+model.name);
            action.Click([this,model](auto const&,auto const&) { if(model.available()) activate(model.id,model.name); else download_repository(model.source.empty()?model.name:model.source); });
            actions.Children().Append(action); content.Children().Append(actions); list.Items().Append(card(content));
        }
        text(L"LibrarySummary",discover_tab_?L"Hugging Face · Prepared Flare cores":std::to_wstring(count)+(count==1?L" core on this PC":L" cores on this PC"));
        control<StackPanel>(L"RecommendedSection").Visibility(discover_tab_&&recommended_count?Visibility::Visible:Visibility::Collapsed);
        control<TextBlock>(L"RecommendedMoreLabel").Visibility(other_count?Visibility::Visible:Visibility::Collapsed);
        control<StackPanel>(L"ModelsEmpty").Visibility(count?Visibility::Collapsed:Visibility::Visible);
        text(L"ModelsEmptyTitle",discover_tab_?L"Find your next local model":L"Your local library is ready");
        text(L"ModelsEmptyDetail",discover_tab_?L"Search Hugging Face for prepared Flare cores, or add a repository directly.":L"Open a downloaded core file or folder to use it in place.");
        resize_cards();
    }
    void resize_cards() {
        if (!root_) return;
        auto list=control<GridView>(L"ModelList");
        auto width=list.ActualWidth();
        if(width<=0) return;
        auto columns=width>=650?2.0:1.0;
        constexpr double gap=14,scrollbar_space=24;
        auto card_width=std::max(220.0,std::floor((width-scrollbar_space-gap*(columns-1))/columns));
        control<StackPanel>(L"RecommendedSection").Width(card_width*columns+gap*(columns-1));
        for(auto const& item:list.Items()) if(auto border=item.try_as<Border>()) { border.Width(card_width); border.Margin(Thickness{0,0,gap,16}); }
    }

    struct DownloadView { uint64_t id; StackPanel panel; TextBlock title, phase, detail; ProgressBar progress; Button pause,cancel,resume,use; std::string installed_model; };
    void update_download_progress(ProgressBar const& progress,flare_download const& value) {
        bool metered=value.transfer_active&&value.total_known&&value.network_total&&value.network_bytes<value.network_total;
        progress.IsIndeterminate(value.state==FLARE_DOWNLOAD_RUNNING&&!metered);
        progress.Maximum(100);
        progress.Value(value.state==FLARE_DOWNLOAD_COMPLETE?100:value.total_known&&value.network_total?std::min(99.0,100.0*static_cast<double>(value.network_bytes)/value.network_total):0);
    }
    void render_downloads(flare_snapshot const* snapshot) {
        auto count = flare_snapshot_download_count(snapshot);
        text(L"DownloadsSummary", count == 0 ? L"No downloads" :
            std::to_wstring(count) + (count==1?L" transfer":L" transfers") + (count > 100 ? L". Showing the first 100." : L"."));
        control<Button>(L"DownloadsFindButton").Visibility(count?Visibility::Collapsed:Visibility::Visible);
        uint32_t running=0;flare_download summary{};int summary_rank=99;
        auto list = control<StackPanel>(L"DownloadList");
        list.Visibility(count?Visibility::Visible:Visibility::Collapsed);
        std::vector<uint64_t> seen;
        for (size_t i = 0; i < std::min(count, size_t{100}); ++i) {
            flare_download value{};
            if (!flare_snapshot_download(snapshot, i, &value)) continue;
            if(value.state==FLARE_DOWNLOAD_RUNNING||value.state==FLARE_DOWNLOAD_QUEUED)++running;
            int rank=value.state==FLARE_DOWNLOAD_RUNNING?0:value.state==FLARE_DOWNLOAD_QUEUED?1:value.state==FLARE_DOWNLOAD_PAUSED?2:value.state==FLARE_DOWNLOAD_FAILED?3:4;
            if(rank<summary_rank){summary_rank=rank;summary=value;}
            seen.push_back(value.operation_id);
            auto found = std::find_if(download_views_.begin(), download_views_.end(), [&](auto const& view) { return view.id == value.operation_id; });
            if (found == download_views_.end()) {
                DownloadView view{ value.operation_id };
                view.panel.Spacing(11);
                view.title.FontSize(17);view.title.FontWeight(Windows::UI::Text::FontWeights::SemiBold());view.phase.FontSize(12);
                view.title.TextWrapping(TextWrapping::Wrap);
                view.phase.TextWrapping(TextWrapping::Wrap);
                view.detail.TextWrapping(TextWrapping::Wrap);
                Grid heading;heading.ColumnSpacing(14);heading.ColumnDefinitions().Append(ColumnDefinition());ColumnDefinition state_column;state_column.Width(GridLength{1,GridUnitType::Auto});heading.ColumnDefinitions().Append(state_column);heading.Children().Append(view.title);Grid::SetColumn(view.phase,1);heading.Children().Append(view.phase);view.panel.Children().Append(heading);
                view.progress.Height(5);view.progress.MinHeight(5);
                view.panel.Children().Append(view.progress);
                view.panel.Children().Append(view.detail);
                StackPanel buttons;
                buttons.Orientation(Orientation::Horizontal);
                buttons.Spacing(10);
                view.cancel.Content(box_value(L"Cancel download"));
                view.pause.Content(box_value(L"Pause"));
                view.resume.Content(box_value(L"Resume"));
                view.cancel.Click([this, id = value.operation_id](auto const&, auto const&) { accepted(flare_app_cancel(app_, id)); });
                view.resume.Click([this, id = value.operation_id](auto const&, auto const&) { accepted(flare_app_resume(app_, id)); });
                view.pause.Click([this,id=value.operation_id](auto const&,auto const&){accepted(flare_app_pause_download(app_,id));});
                view.use.Content(box_value(L"Run model"));view.use.Click([this,id=value.operation_id](auto const&,auto const&){for(auto const& item:download_views_)if(item.id==id&&!item.installed_model.empty()){activate(item.installed_model,std::wstring(item.title.Text()));break;}});
                buttons.Children().Append(view.pause);
                buttons.Children().Append(view.cancel);
                buttons.Children().Append(view.resume);
                buttons.Children().Append(view.use);
                view.panel.Children().Append(buttons);
                list.Children().Append(card(view.panel));
                download_views_.push_back(view);
                found = std::prev(download_views_.end());
            }
            found->title.Text(wide(value.name));
            found->installed_model=utf8(value.installed_model);
            Automation::AutomationProperties::SetName(found->cancel, L"Cancel download of " + wide(value.name));
            Automation::AutomationProperties::SetName(found->resume, L"Resume download of " + wide(value.name));
            auto phase = value.state == FLARE_DOWNLOAD_COMPLETE ? L"Complete" : value.state == FLARE_DOWNLOAD_FAILED ? L"Failed" :
                value.state == FLARE_DOWNLOAD_CANCELLED ? L"Cancelled" : value.state == FLARE_DOWNLOAD_PAUSED ? L"Paused" :
                value.state == FLARE_DOWNLOAD_QUEUED ? L"Queued" : L"Downloading";
            found->phase.Text(value.phase.length&&value.state==FLARE_DOWNLOAD_RUNNING?wide(value.phase):std::wstring(phase));
            update_download_progress(found->progress,value);
            bool transferring=value.state==FLARE_DOWNLOAD_RUNNING&&value.transfer_active;
            auto detail = bytes(value.network_bytes) + (value.total_known&&(transferring||value.state!=FLARE_DOWNLOAD_RUNNING) ? L" / " + bytes(value.network_total) : L" transferred");
            if(value.state==FLARE_DOWNLOAD_RUNNING&&value.planning_total_known)detail=L"File metadata: "+std::to_wstring(value.planning_completed)+L" / "+std::to_wstring(value.planning_total)+L" checked · "+detail;
            if (transferring&&std::isfinite(value.bytes_per_second)&&value.bytes_per_second>0) {
                detail+=L" · "+bytes(static_cast<uint64_t>(value.bytes_per_second))+L"/s";
                if(value.total_known&&value.network_total>value.network_bytes)detail+=L" · about "+number((value.network_total-value.network_bytes)/value.bytes_per_second,0)+L" s remaining";
            }
            if(!value.total_known&&!value.network_bytes&&value.state==FLARE_DOWNLOAD_COMPLETE)detail=L"Core installed · transfer totals not recorded for this entry";
            if(!value.total_known&&!value.network_bytes&&value.state==FLARE_DOWNLOAD_PAUSED)detail=L"Any partial data is kept for resume. Transfer totals are not available.";
            if (value.verified_bytes) detail += L" · " + bytes(value.verified_bytes) + L" verified";
            if (value.materialized_bytes) detail += L" · " + bytes(value.materialized_bytes) + L" installed";
            if (value.error.length) detail += L"\n" + wide(value.error);
            found->detail.Text(detail);
            found->cancel.Visibility(value.can_cancel ? Visibility::Visible : Visibility::Collapsed);
            found->resume.Visibility(value.can_resume ? Visibility::Visible : Visibility::Collapsed);
            found->pause.Visibility(value.can_cancel ? Visibility::Visible : Visibility::Collapsed);
            found->use.Visibility(value.state==FLARE_DOWNLOAD_COMPLETE&&value.installed_model.length?Visibility::Visible:Visibility::Collapsed);
        }
        control<InfoBadge>(L"DownloadBadge").Value(static_cast<int32_t>(running));control<InfoBadge>(L"DownloadBadge").Visibility(running?Visibility::Visible:Visibility::Collapsed);
        text(L"DownloadSummary",count?wide(summary.name)+L" · "+wide(summary.phase):L"No downloads queued");
        control<Grid>(L"HomeSupportPanels").Visibility(dashboard_.available||running?Visibility::Visible:Visibility::Collapsed);
        control<Border>(L"DownloadSummaryCard").Visibility(running?Visibility::Visible:Visibility::Collapsed);
        Grid::SetColumnSpan(control<Border>(L"MemoryCard"),running?1:2);
        auto summary_progress=control<ProgressBar>(L"DownloadSummaryProgress");summary_progress.Visibility(count?Visibility::Visible:Visibility::Collapsed);update_download_progress(summary_progress,summary);
        if (std::any_of(download_views_.begin(), download_views_.end(), [&](auto const& view) { return std::find(seen.begin(), seen.end(), view.id) == seen.end(); })) {
            download_views_.clear();
            list.Children().Clear();
            render_downloads(snapshot);
        }
    }

    fire_and_forget review_trust(uint64_t id, std::string publisher, std::wstring content) {
        auto lifetime = get_strong();
        ModalScope modal_scope{modal_open_};if(!modal_scope)co_return;
        try {
            ContentDialog dialog;
            dialog.XamlRoot(root_.XamlRoot());
            dialog.Title(box_value(L"Review model publisher"));
            dialog.Content(box_value(content));
            dialog.PrimaryButtonText(L"Trust this publisher and continue");
            dialog.CloseButtonText(L"Cancel loading");
            dialog.DefaultButton(ContentDialogButton::Close);
            auto result = co_await dialog.ShowAsync();
            if (!closing_) accepted(result == ContentDialogResult::Primary ? flare_app_accept_trust(app_, id, publisher.data(), publisher.size()) : flare_app_reject_trust(app_, id));
        } catch (hresult_error const& error) { show_error(std::wstring(error.message())); }
        trust_dialog_id_ = 0;
    }
    void begin_close() {
        if (closing_) return;
        closing_ = true;
        flare_app_close(app_);
        control<NavigationView>(L"Navigation").IsEnabled(false);
        text(L"AppTitle", L"Flare — Closing…");
    }
    void poll() {
        try {
            if (closing_) {
                if (flare_app_is_closed(app_)) {
                    timer_.Stop();
                    flare_app_destroy(app_);
                    app_ = nullptr;
                    closed_ = true;
                    clear_taskbar_properties();
                    window_.Close();
                }
                return;
            }
            Snapshot snapshot(flare_app_snapshot(app_), flare_snapshot_destroy);
            if (!snapshot) { show_error(wide(flare_last_error())); return; }
            flare_status status{};
            if (!flare_snapshot_status(snapshot.get(), &status)) { show_error(wide(flare_last_error())); return; }
            if(!control<InfoBar>(L"Notice").IsOpen())control<InfoBar>(L"Notice").Margin(Thickness{0,0,0,0});
            render_status(status);
            render_dashboard(snapshot.get(),status);
            update_runtime_settings(snapshot.get(),status);
            auto error = wide(status.error);
            if (!error.empty() && error != last_error_) show_error(error);
            last_error_ = std::move(error);
            render_models(snapshot.get());
            render_downloads(snapshot.get());
            update_hardware(snapshot.get(),status);
            update_detail(snapshot.get(),status);
            update_chat(snapshot.get(),status);
            flare_trust_review review{};
            if (!modal_open_ && !trust_dialog_id_ && flare_snapshot_trust(snapshot.get(), &review)) {
                trust_dialog_id_ = review.decision_id;
                auto content = L"Publisher: " + wide(review.publisher) + L"\n\nFingerprint: " + wide(review.fingerprint) +
                    L"\n\nIdentity: " + wide(review.identity_status) + L"\n\nOfficial source: " + wide(review.official_source) +
                    L"\n\nLicense: " + wide(review.license) + L"\n\nTrust only a publisher whose identity and source you have verified.";
                review_trust(review.decision_id, utf8(review.publisher), content);
            }
        } catch (hresult_error const& error) { show_error(std::wstring(error.message())); }
        catch (std::exception const& error) { show_error(std::wstring(to_hstring(error.what()))); }
    }

    #include "Details.inl"
    #include "Dashboard.inl"
    #include "Settings.inl"
    #include "Chat.inl"

    std::wstring config_, endpoint_, last_error_, model_fingerprint_, page_, model_directory_, active_name_, pending_model_name_, detail_fingerprint_,home_model_fingerprint_;
    std::wstring desktop_launcher_;
    std::string pending_model_id_;
    std::wstring copied_endpoint_;
    std::chrono::steady_clock::time_point endpoint_copy_until_{};
    std::wstring loading_fingerprint_;

    ResourceDictionary theme_resources_{nullptr};
    std::vector<ModelItem> models_;
    flare_dashboard dashboard_{};
    std::vector<flare_dashboard_point> dashboard_points_;
    std::vector<std::wstring> observed_steps_;
    std::unordered_map<std::string,uint32_t> compatibilities_;
    double card_padding_{16},chat_last_offset_{};
    uint64_t active_requests_{},waiting_requests_{};
    uint64_t chat_operation_{},chat_session_{UINT64_MAX},chat_conversation_{UINT64_MAX};
    std::wstring chat_error_,chat_submitted_text_;
    std::vector<ChatRow> chat_rows_;
    bool chat_ready_{},chat_busy_{},chat_submit_pending_{},chat_stop_pending_{},chat_composing_{};
    bool chat_follow_{true},chat_scroll_pending_{},chat_auto_scrolling_{};
    bool modal_open_{},applying_preferences_{},preferences_pending_{},preferences_initialized_{},pending_compact_{},persisted_compact_{};
    bool discover_tab_{true},catalog_requested_{},filter_incompatible_{},detail_open_{};
    bool home_inventory_pending_{true};
    bool runtime_settings_initialized_{},runtime_settings_applying_{},runtime_settings_dirty_{},runtime_settings_busy_{},runtime_reload_pending_{};
    uint64_t runtime_settings_operation_{};
    std::wstring runtime_settings_error_,runtime_settings_saved_form_;
    ContentDialog detail_dialog_{nullptr};
    StackPanel detail_rows_{nullptr};
    ModelItem detail_model_;
    XamlTypeInfo::XamlControlsXamlMetaDataProvider metadata_;
    Window window_{ nullptr };
    FrameworkElement root_{ nullptr };
    Microsoft::UI::Dispatching::DispatcherQueueTimer timer_{ nullptr };
    HWND hwnd_{};
    com_ptr<ITaskbarList3> taskbar_;
    com_ptr<IPropertyStore> taskbar_properties_;
    flare_app* app_{};
    uint32_t session_state_{};
    uint64_t trust_dialog_id_{};
    uint64_t current_operation_{};
    bool closing_{}, closed_{}, can_change_model_{true};
    std::vector<DownloadView> download_views_;
};
}

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    try {
        int count{};
        auto arguments = CommandLineToArgvW(GetCommandLineW(), &count);
        if (!arguments) throw_last_error();
        std::unique_ptr<wchar_t*, decltype(&LocalFree)> owner(arguments, LocalFree);
        std::wstring config,desktop_launcher;
        for (int i = 1; i < count; ++i) {
            if (std::wstring_view(arguments[i]) == L"--config" && i + 1 < count) config = arguments[++i];
            else if(std::wstring_view(arguments[i])==L"--desktop-launcher"&&i+1<count){
                desktop_launcher=arguments[++i];
                if(desktop_launcher.empty())throw hresult_invalid_argument(L"Desktop launcher path is empty");
            }
            else throw hresult_invalid_argument(L"Usage: flare-winui.exe [--config PATH]");
        }
        if(!config.empty())config=absolute_windows_path(config);
        if(!desktop_launcher.empty())desktop_launcher=absolute_windows_path(desktop_launcher);
        auto identity=SetCurrentProcessExplicitAppUserModelID(desktop_app_id);
        if(FAILED(identity))std::cerr<<"Taskbar process identity unavailable: "<<identity<<std::endl;
        if (!flare_runtime_initialize()) throw hresult_error(E_FAIL, hstring(wide(flare_last_error())));
        init_apartment(apartment_type::single_threaded);
        Application::Start([config,desktop_launcher](auto const&) { make<DesktopApp>(config,desktop_launcher); });
        return application_exit_code;
    } catch (hresult_error const& error) {
        std::cerr << to_string(error.message()) << std::endl;
        MessageBoxW(nullptr, error.message().c_str(), L"Flare could not start", MB_OK | MB_ICONERROR);
        return 1;
    } catch (std::exception const& error) {
        std::cerr << error.what() << std::endl;
        auto message = to_hstring(error.what());
        MessageBoxW(nullptr, message.c_str(), L"Flare could not start", MB_OK | MB_ICONERROR);
        return 1;
    }
}
